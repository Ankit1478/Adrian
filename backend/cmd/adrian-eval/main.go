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
	"os"

	"github.com/secureagentics/Adrian/backend/internal/engine"
)

func main() {
	var (
		casesPath = flag.String("cases", "cmd/adrian-eval/testdata/cases.jsonl", "JSONL file of labelled cases")
		url       = flag.String("url", os.Getenv("ADRIAN_LLM_URL"), "judge endpoint (default $ADRIAN_LLM_URL)")
		key       = flag.String("key", os.Getenv("ADRIAN_LLM_API_KEY"), "judge API key (default $ADRIAN_LLM_API_KEY)")
		model     = flag.String("model", os.Getenv("ADRIAN_LLM_MODEL"), "judge model name (default $ADRIAN_LLM_MODEL)")
		outPath   = flag.String("out", "", "also write the full report as JSON to this file")
		runs      = flag.Int("runs", 1, "run every case this many times and report averages and unstable cases")
	)
	flag.Parse()

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

	judge := engine.NewHTTPClient(*url, *key, *model, nil, nil)

	// One run keeps the detailed report. Several runs print a summary of
	// each, the averages, and the cases whose answer changed.
	var output any
	if *runs == 1 {
		report := Score(Run(context.Background(), judge, cases))
		report.Print(os.Stdout)
		output = report
	} else {
		reps := make([]Report, 0, *runs)
		for i := 1; i <= *runs; i++ {
			fmt.Fprintf(os.Stderr, "run %d of %d...\n", i, *runs)
			reps = append(reps, Score(Run(context.Background(), judge, cases)))
		}
		multi := Summarise(reps)
		multi.Print(os.Stdout)
		output = multi
	}

	if *outPath != "" {
		data, err := json.MarshalIndent(output, "", "  ")
		if err == nil {
			err = os.WriteFile(*outPath, data, 0o644)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("\nFull report written to %s\n", *outPath)
	}
}
