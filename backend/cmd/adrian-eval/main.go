// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SecureAgentics

// Command adrian-eval scores the LLM judge against a file of labelled
// cases. It calls the same engine.Classifier the backend uses, so the
// prompt, trace rendering and M-code parsing are exactly the production
// path. Each case is judged on its own, with no history and no agent
// profile.
//
// Usage, from the backend directory:
//
//	ADRIAN_LLM_URL=http://localhost:8081/v1/chat/completions \
//	ADRIAN_LLM_API_KEY=... ADRIAN_LLM_MODEL=... \
//	go run ./cmd/adrian-eval -cases cmd/adrian-eval/testdata/cases.jsonl
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/secureagentics/Adrian/backend/internal/engine"
	"github.com/secureagentics/Adrian/backend/internal/store"
)

// checkProfileNames fails loudly when a case names a profile that does
// not exist, instead of silently judging it against the generic remit.
func checkProfileNames(cases []Case, profiles map[string]Profile) error {
	for _, c := range cases {
		if c.Profile == "" {
			continue
		}
		if _, ok := profiles[c.Profile]; !ok {
			return fmt.Errorf("case %s names unknown profile %q", c.ID, c.Profile)
		}
	}
	return nil
}

func main() {
	var (
		casesPath = flag.String("cases", "cmd/adrian-eval/testdata/cases.jsonl", "JSONL file of labelled cases")
		url       = flag.String("url", os.Getenv("ADRIAN_LLM_URL"), "judge endpoint (default $ADRIAN_LLM_URL)")
		key       = flag.String("key", os.Getenv("ADRIAN_LLM_API_KEY"), "judge API key (default $ADRIAN_LLM_API_KEY)")
		model     = flag.String("model", os.Getenv("ADRIAN_LLM_MODEL"), "judge model name (default $ADRIAN_LLM_MODEL)")
		outPath   = flag.String("out", "", "save the full report to this file (default: a new file in -results-dir)")
		resultDir = flag.String("results-dir", "eval-results", "folder for automatically saved reports")
		noSave    = flag.Bool("no-save", false, "do not save the report, only print it")
		runs      = flag.Int("runs", 1, "run every case this many times and report averages and unstable cases")
		workers   = flag.Int("concurrency", 4, "how many cases to judge at once")
		retries   = flag.Int("retries", 2, "extra attempts after a network or server error (judge answers are never retried)")
		backoff   = flag.Duration("backoff", 2*time.Second, "wait before the first retry; doubles each time")
		gate      = Thresholds{}
		compare   = flag.Bool("compare", false, "compare two saved reports: adrian-eval -compare old.json new.json")
		failRegr  = flag.Bool("fail-on-regression", false, "with -compare, exit 1 if the new report is worse")
		priceIn   = flag.Float64("price-in", 0, "USD per million input tokens, for the cost estimate")
		priceOut  = flag.Float64("price-out", 0, "USD per million output tokens, for the cost estimate")
		priceHit  = flag.Float64("price-cached", 0, "USD per million cached input tokens (defaults to -price-in)")
		promptDir = flag.String("prompt-dir", "", "folder holding system_prompt.md and few_shot.md, for the prompt fingerprint")
		profPath  = flag.String("profiles", "cmd/adrian-eval/testdata/profiles.json", "JSON file of agent profiles a case can name")
	)
	flag.Float64Var(&gate.MaxMissedBlocksPct, "max-missed-blocks", -1, "fail if missed blocks exceed this percent (off if negative)")
	flag.Float64Var(&gate.MaxFalsePositivesPct, "max-false-positives", -1, "fail if false positives exceed this percent (off if negative)")
	flag.Float64Var(&gate.MinTierAccuracyPct, "min-tier-accuracy", -1, "fail if tier accuracy is below this percent (off if negative)")
	flag.Float64Var(&gate.MaxErrorPct, "max-errors", -1, "fail if errors exceed this percent of cases (off if negative)")
	flag.Parse()

	if *compare {
		if flag.NArg() != 2 {
			fmt.Fprintln(os.Stderr, "usage: adrian-eval -compare old.json new.json")
			os.Exit(2)
		}
		old, err := LoadSnapshot(flag.Arg(0))
		if err == nil {
			var cur Snapshot
			if cur, err = LoadSnapshot(flag.Arg(1)); err == nil {
				c := Compare(old, cur)
				c.Print(os.Stdout)
				if *failRegr && c.Regressed() {
					os.Exit(1)
				}
				return
			}
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if *runs < 1 {
		fmt.Fprintln(os.Stderr, "-runs must be 1 or more")
		os.Exit(2)
	}
	if *url == "" || *model == "" {
		fmt.Fprintln(os.Stderr, "need a judge: set -url and -model (or ADRIAN_LLM_URL and ADRIAN_LLM_MODEL)")
		os.Exit(2)
	}

	f, err := os.Open(*casesPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	cases, err := LoadCases(f)
	f.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", *casesPath, err)
		os.Exit(2)
	}

	// Count tokens from every model response. The judge client uses the
	// default transport, so wrapping it covers all of its calls.
	meter := &usageMeter{base: http.DefaultTransport}
	http.DefaultTransport = meter
	prices := Prices{Input: *priceIn, CachedInput: *priceHit, Output: *priceOut}

	// A sliding window gives each case its own conversation, keyed by the
	// case id, so the steps of a multi-step case chain through history
	// exactly as they do in production while separate cases stay
	// isolated. Single-step cases simply see an empty history.
	window := engine.NewSlidingWindow(engine.WindowOpts{Size: 16, TTL: time.Hour})

	// Cases that name a profile are judged against that customer's remit,
	// resolved from a temporary database through the production path.
	// A missing profiles file is fine: those cases use the generic remit.
	profiles, err := LoadProfiles(*profPath)
	if err != nil && !os.IsNotExist(err) {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	profStore, err := NewProfileStore(context.Background(), profiles)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer profStore.Close()
	if err := checkProfileNames(cases, profiles); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	var judgeStore *store.Store
	if profStore != nil {
		judgeStore = profStore.Store
	}
	judge := engine.NewHTTPClient(*url, *key, *model, window, judgeStore)
	opt := Options{Concurrency: *workers, Retries: *retries, Backoff: *backoff, Profiles: profStore}

	started := time.Now()
	reps := make([]Report, 0, *runs)
	for i := 1; i <= *runs; i++ {
		if *runs > 1 {
			fmt.Fprintf(os.Stderr, "run %d of %d...\n", i, *runs)
		}
		before := meter.Snapshot()
		report := Score(RunWith(context.Background(), judge, cases, opt))
		used := meter.Snapshot().Sub(before).WithCost(prices)
		report.Usage = &used
		reps = append(reps, report)
	}

	commit, dirty := gitState()
	casesSum, _ := fileSHA256(*casesPath)
	meta := &Meta{
		Tool: "adrian-eval", StartedAt: started.UTC().Format(time.RFC3339),
		DurationSec: float64(time.Since(started).Milliseconds()) / 1000,
		Model:       *model, Endpoint: cleanEndpoint(*url),
		Runs: *runs, Concurrency: *workers, Retries: *retries,
		CasesFile: *casesPath, CasesSHA256: casesSum, CaseCount: len(cases),
		PromptSHA256: promptSHA256(*promptDir), GitCommit: commit, GitDirty: dirty,
		Backoff:            backoff.String(),
		OmitSamplingParams: envTrue("ADRIAN_LLM_OMIT_SAMPLING_PARAMS"),
	}
	if prices.Known() {
		meta.Prices = &prices
	}

	// One run keeps the detailed report. Several runs print a summary of
	// each, the averages, and the cases whose answer changed.
	var output any
	var verdict *GateResult
	if *runs == 1 {
		report := reps[0]
		report.Meta = meta
		report.Print(os.Stdout)
		if !gate.Off() {
			g := CheckGate(report.gateMetrics(), gate)
			report.Gate, verdict = &g, &g
		}
		output = report
	} else {
		multi := Summarise(reps)
		multi.Meta = meta
		total := meter.Snapshot().WithCost(prices)
		multi.Usage = &total
		multi.Print(os.Stdout)
		if !gate.Off() {
			g := CheckGate(multi.gateMetrics(), gate)
			multi.Gate, verdict = &g, &g
		}
		output = multi
	}
	if verdict != nil {
		verdict.Print(os.Stdout)
	}

	// Every run is saved unless -no-save is given, so a result is never
	// lost for want of -out.
	savePath := *outPath
	if savePath == "" && !*noSave {
		if err := ensureResultsDir(*resultDir); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		savePath = autoSavePath(*resultDir, *model, *runs, started)
	}
	if savePath != "" {
		// Create the folder for -out too, so a missing folder never loses
		// a finished (and paid-for) run.
		if dir := filepath.Dir(savePath); dir != "." {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
		data, err := json.MarshalIndent(output, "", "  ")
		if err == nil {
			err = os.WriteFile(savePath, data, 0o644)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("\nFull report saved to %s\n", savePath)
	}
	if verdict != nil && !verdict.Passed {
		os.Exit(1)
	}
}
