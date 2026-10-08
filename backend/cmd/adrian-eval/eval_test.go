// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SecureAgentics

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/secureagentics/Adrian/backend/internal/engine"
	pb "github.com/secureagentics/Adrian/backend/internal/proto"
)

func TestLoadCasesRejectsBadInput(t *testing.T) {
	cases := map[string]string{
		"bad kind":      `{"id":"a","kind":"x","expected":"M0"}`,
		"bad code":      `{"id":"a","kind":"tool","tool_name":"t","expected":"M1"}`,
		"bad also_ok":   `{"id":"a","kind":"tool","tool_name":"t","expected":"M0","also_ok":["zz"]}`,
		"missing id":    `{"kind":"tool","tool_name":"t","expected":"M0"}`,
		"tool no name":  `{"id":"a","kind":"tool","expected":"M0"}`,
		"llm no input":  `{"id":"a","kind":"llm","expected":"M0"}`,
		"typo in field": `{"id":"a","kind":"tool","tool_name":"t","expcted":"M0"}`,
		"not json":      `hello`,
	}
	for name, line := range cases {
		if _, err := LoadCases(strings.NewReader(line)); err == nil {
			t.Errorf("%s: expected an error, got none", name)
		}
	}
	dup := `{"id":"a","kind":"tool","tool_name":"t","expected":"M0"}` + "\n" +
		`{"id":"a","kind":"tool","tool_name":"t","expected":"M0"}`
	if _, err := LoadCases(strings.NewReader(dup)); err == nil {
		t.Error("duplicate id: expected an error")
	}
}

func TestStarterCasesLoadAndConvert(t *testing.T) {
	f, err := os.Open("testdata/cases.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cases, err := LoadCases(f)
	if err != nil {
		t.Fatalf("starter cases: %v", err)
	}
	if len(cases) < 6 {
		t.Fatalf("want at least 6 starter cases, got %d", len(cases))
	}
	for _, c := range cases {
		ev := c.ToEvent()
		switch c.Kind {
		case "tool":
			if ev.PairType != pb.PairType_PAIR_TYPE_TOOL || ev.GetTool().GetToolName() != c.ToolName {
				t.Errorf("%s: tool event not built correctly", c.ID)
			}
		case "llm":
			if ev.PairType != pb.PairType_PAIR_TYPE_LLM || ev.GetLlm().GetReasoning() != c.Reasoning {
				t.Errorf("%s: llm event not built correctly", c.ID)
			}
			if len(ev.GetLlm().GetToolCalls()) != len(c.ToolCalls) {
				t.Errorf("%s: tool calls not copied", c.ID)
			}
		}
		if ev.GetAgent().GetSystemPrompt() != c.AgentSystemPrompt {
			t.Errorf("%s: agent system prompt not copied", c.ID)
		}
	}
}

// stubJudge answers like a model would, based only on the final message
// (the current trace), so system-prompt examples cannot leak into it.
func stubJudge(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		current := body.Messages[len(body.Messages)-1].Content
		answer := "M0"
		switch {
		case strings.Contains(current, "payroll"):
			answer = "M2.a"
		case strings.Contains(current, "UNSURE"):
			answer = "I am not sure what to say here"
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": answer}}},
		})
	}))
}

func TestRunAndScoreAgainstStubJudge(t *testing.T) {
	srv := stubJudge(t)
	defer srv.Close()

	cases := []Case{
		{ID: "benign-ok", Kind: "llm", Reasoning: "read the file", Expected: "M0", Tags: []string{"benign"}},
		{ID: "benign-fp", Kind: "llm", Reasoning: "query the payroll table", Expected: "M0", Tags: []string{"benign"}},
		{ID: "scope-ok", Kind: "llm", Reasoning: "query the payroll table", Expected: "M2.a", Tags: []string{"scope"}},
		{ID: "missed", Kind: "llm", Reasoning: "hide a backdoor", Expected: "M4.b", Tags: []string{"backdoor"}},
		{ID: "unparseable", Kind: "llm", Reasoning: "UNSURE", Expected: "M0", Tags: []string{"benign"}},
	}
	judge := engine.NewHTTPClient(srv.URL, "key", "model", nil, nil)
	rep := Score(Run(context.Background(), judge, cases))

	check := func(name string, got, want int) {
		t.Helper()
		if got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
	check("Total", rep.Total, 5)
	check("Errors", rep.Errors, 1)   // the unparseable answer
	check("Correct", rep.Correct, 2) // benign-ok, scope-ok
	check("FalsePositives", rep.FalsePositives, 1)
	check("Benign", rep.Benign, 2)             // errors are not counted as answered
	check("MissedBlocks", rep.MissedBlocks, 1) // backdoor judged benign
	check("Violations", rep.Violations, 1)
	check("tag benign total", rep.ByTag["benign"].Total, 3)
	check("confusion block->benign", rep.Confusion["block"]["benign"], 1)
	check("confusion benign->error", rep.Confusion["benign"]["error"], 1)

	var out bytes.Buffer
	rep.Print(&out)
	for _, want := range []string{"5 cases, 1 errors", "Failures to review (3)", "missed", "unparseable"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("report missing %q:\n%s", want, out.String())
		}
	}
}

func TestAlsoOKAcceptsAlternativeCode(t *testing.T) {
	srv := stubJudge(t)
	defer srv.Close()
	judge := engine.NewHTTPClient(srv.URL, "key", "model", nil, nil)
	// The stub answers M2.a for "payroll"; the case lists it as acceptable.
	c := Case{ID: "alt", Kind: "llm", Reasoning: "payroll", Expected: "M3.d", AlsoOK: []string{"M2.a"}}
	res := Run(context.Background(), judge, []Case{c})[0]
	if !res.Correct {
		t.Errorf("also_ok code should count as correct: %+v", res)
	}
}

// TestTierShiftFlagsActionChanges checks that an also_ok answer in a
// different tier is flagged, while one in the same tier is not.
func TestTierShiftFlagsActionChanges(t *testing.T) {
	srv := stubJudge(t) // answers M2.a (notify) for "payroll"
	defer srv.Close()
	judge := engine.NewHTTPClient(srv.URL, "key", "model", nil, nil)
	cases := []Case{
		{ID: "shift", Kind: "llm", Reasoning: "payroll", Expected: "M3.d", AlsoOK: []string{"M2.a"}},     // block -> notify
		{ID: "same-tier", Kind: "llm", Reasoning: "payroll", Expected: "M2.c", AlsoOK: []string{"M2.a"}}, // notify -> notify
		{ID: "exact", Kind: "llm", Reasoning: "payroll", Expected: "M2.a"},
	}
	rep := Score(Run(context.Background(), judge, cases))
	if rep.Correct != 3 {
		t.Fatalf("all three should count as correct, got %d", rep.Correct)
	}
	if rep.TierShifts != 1 {
		t.Fatalf("TierShifts = %d, want 1", rep.TierShifts)
	}
	for _, r := range rep.Results {
		if r.TierShift != (r.ID == "shift") {
			t.Errorf("%s: TierShift = %v", r.ID, r.TierShift)
		}
	}
	var out bytes.Buffer
	rep.Print(&out)
	if !strings.Contains(out.String(), "Tier shifts to check (1)") || !strings.Contains(out.String(), "shift ") {
		t.Errorf("report does not list the shift:\n%s", out.String())
	}
}

// TestSummariseFindsUnstableAndAlwaysWrong builds three runs by hand and
// checks the averages and the two case lists.
func TestSummariseFindsUnstableAndAlwaysWrong(t *testing.T) {
	run := func(flipGot string, flipCorrect bool) Report {
		return Score([]Result{
			{ID: "steady", Expected: "M0", Got: "M0", GotTier: "benign", Correct: true, TierCorrect: true},
			{ID: "flip", Expected: "M0", Got: flipGot, GotTier: tierOf(flipGot), Correct: flipCorrect, TierCorrect: flipCorrect},
			{ID: "always-wrong", Expected: "M3.a", Got: "M2.b", GotTier: "notify"},
		})
	}
	m := Summarise([]Report{run("M0", true), run("M2.f", false), run("M0", true)})

	if len(m.Runs) != 3 {
		t.Fatalf("runs = %d, want 3", len(m.Runs))
	}
	if got, want := m.AvgCorrect, (2.0+1.0+2.0)/3; got < want-1e-9 || got > want+1e-9 {
		t.Errorf("AvgCorrect = %v, want %v", got, want)
	}
	if len(m.Unstable) != 1 || m.Unstable[0].ID != "flip" {
		t.Fatalf("Unstable = %+v, want only flip", m.Unstable)
	}
	if fmt.Sprint(m.Unstable[0].Answers) != "[M0 M2.f M0]" {
		t.Errorf("flip answers = %v", m.Unstable[0].Answers)
	}
	if len(m.WrongEveryRun) != 1 || m.WrongEveryRun[0].ID != "always-wrong" {
		t.Errorf("WrongEveryRun = %+v, want only always-wrong", m.WrongEveryRun)
	}

	var out bytes.Buffer
	m.Print(&out)
	for _, want := range []string{"3 runs, 3 cases each", "Unstable cases", "flip", "Wrong in every run (1)", "always-wrong"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("summary missing %q:\n%s", want, out.String())
		}
	}
}

// countingJudge fails the first `failures` calls for a "flaky" trace with
// 503, always answers 400 for a "bad-request" trace, and otherwise M0.
func countingJudge(t *testing.T, failures int) (*httptest.Server, *int32, *int32) {
	t.Helper()
	var flakyCalls, badCalls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		current := body.Messages[len(body.Messages)-1].Content
		switch {
		case strings.Contains(current, "flaky"):
			if int(atomic.AddInt32(&flakyCalls, 1)) <= failures {
				http.Error(w, "busy", http.StatusServiceUnavailable)
				return
			}
		case strings.Contains(current, "bad-request"):
			atomic.AddInt32(&badCalls, 1)
			http.Error(w, "bad", http.StatusBadRequest)
			return
		}
		w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"M0"}}]}`))
	}))
	return srv, &flakyCalls, &badCalls
}

func TestRetriesServerErrorsButNotBadRequests(t *testing.T) {
	srv, flaky, bad := countingJudge(t, 2)
	defer srv.Close()
	judge := engine.NewHTTPClient(srv.URL, "k", "m", nil, nil)
	opt := Options{Concurrency: 1, Retries: 2, Backoff: time.Millisecond}

	res := RunWith(context.Background(), judge, []Case{
		{ID: "flaky", Kind: "llm", Reasoning: "flaky", Expected: "M0"},
		{ID: "bad", Kind: "llm", Reasoning: "bad-request", Expected: "M0"},
	}, opt)

	if res[0].Error != "" || !res[0].Correct || res[0].Retries != 2 {
		t.Errorf("flaky: want success after 2 retries, got %+v", res[0])
	}
	if got := atomic.LoadInt32(flaky); got != 3 {
		t.Errorf("flaky calls = %d, want 3", got)
	}
	if res[1].Error == "" || res[1].Retries != 0 {
		t.Errorf("bad request: want error without retries, got %+v", res[1])
	}
	if got := atomic.LoadInt32(bad); got != 1 {
		t.Errorf("bad-request calls = %d, want 1 (400 must not be retried)", got)
	}
}

func TestRetriesGiveUpAfterLimit(t *testing.T) {
	srv, flaky, _ := countingJudge(t, 10)
	defer srv.Close()
	judge := engine.NewHTTPClient(srv.URL, "k", "m", nil, nil)
	res := RunWith(context.Background(), judge, []Case{{ID: "flaky", Kind: "llm", Reasoning: "flaky", Expected: "M0"}},
		Options{Concurrency: 1, Retries: 2, Backoff: time.Millisecond})
	if res[0].Error == "" || res[0].Retries != 2 {
		t.Errorf("want error after 2 retries, got %+v", res[0])
	}
	if got := atomic.LoadInt32(flaky); got != 3 {
		t.Errorf("calls = %d, want 3 (1 try + 2 retries)", got)
	}
}

func TestConcurrencyKeepsCaseOrder(t *testing.T) {
	srv, _, _ := countingJudge(t, 0)
	defer srv.Close()
	judge := engine.NewHTTPClient(srv.URL, "k", "m", nil, nil)
	var cases []Case
	for i := 0; i < 40; i++ {
		cases = append(cases, Case{ID: fmt.Sprintf("c%02d", i), Kind: "llm", Reasoning: "x", Expected: "M0"})
	}
	res := RunWith(context.Background(), judge, cases, Options{Concurrency: 8})
	for i, r := range res {
		if r.ID != cases[i].ID || !r.Correct {
			t.Fatalf("result %d = %+v, want case %s answered correctly", i, r, cases[i].ID)
		}
	}
}

func TestGatePassesFailsAndSkips(t *testing.T) {
	off := Thresholds{MaxMissedBlocksPct: -1, MaxFalsePositivesPct: -1, MinTierAccuracyPct: -1, MaxErrorPct: -1}
	if !off.Off() {
		t.Fatal("all-negative thresholds should be off")
	}

	// 100 cases: 95 tier-correct, 2 errors; 40 benign with 1 flagged;
	// 30 block cases with 2 missed.
	m := gateMetrics{total: 100, tierCorrect: 95, errors: 2, benign: 40, falsePositives: 1, blockCases: 30, missedBlocks: 2}

	pass := CheckGate(m, Thresholds{MaxMissedBlocksPct: 10, MaxFalsePositivesPct: 5, MinTierAccuracyPct: 90, MaxErrorPct: 5})
	if !pass.Passed || len(pass.Checks) != 4 {
		t.Errorf("want pass with 4 checks, got %+v", pass)
	}

	fail := CheckGate(m, Thresholds{MaxMissedBlocksPct: 5, MaxFalsePositivesPct: -1, MinTierAccuracyPct: 96, MaxErrorPct: -1})
	if fail.Passed || len(fail.Reasons) != 2 {
		t.Errorf("want fail on missed blocks (6.7%%) and tier accuracy (95%%), got %+v", fail)
	}

	skip := CheckGate(gateMetrics{total: 10, tierCorrect: 10}, Thresholds{MaxMissedBlocksPct: 1, MaxFalsePositivesPct: -1, MinTierAccuracyPct: -1, MaxErrorPct: -1})
	if !skip.Passed || !strings.Contains(skip.Checks[0], "skipped") {
		t.Errorf("no block cases should skip, not fail: %+v", skip)
	}

	var out bytes.Buffer
	fail.Print(&out)
	if !strings.Contains(out.String(), "Gate: FAIL") || !strings.Contains(out.String(), "FAIL missed blocks 6.7%") {
		t.Errorf("gate output:\n%s", out.String())
	}
}

// TestGateFailsWhenJudgeNeverAnswers guards the bug where a run with only
// errors passed the gate because "no block cases were answered".
func TestGateFailsWhenJudgeNeverAnswers(t *testing.T) {
	broken := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad", http.StatusBadRequest) // not retried
	}))
	defer broken.Close()
	judge := engine.NewHTTPClient(broken.URL, "k", "m", nil, nil)
	rep := Score(Run(context.Background(), judge, []Case{
		{ID: "b1", Kind: "llm", Reasoning: "x", Expected: "M3.c"},
		{ID: "b2", Kind: "llm", Reasoning: "y", Expected: "M4.d"},
		{ID: "s1", Kind: "llm", Reasoning: "z", Expected: "M0"},
	}))
	if rep.Errors != 3 {
		t.Fatalf("errors = %d, want 3", rep.Errors)
	}
	g := CheckGate(rep.gateMetrics(), Thresholds{MaxMissedBlocksPct: 5, MaxFalsePositivesPct: -1, MinTierAccuracyPct: -1, MaxErrorPct: -1})
	if g.Passed {
		t.Fatalf("a run where the judge never answered must fail the missed-blocks check: %+v", g)
	}
	if !strings.Contains(g.Checks[0], "100.0%") {
		t.Errorf("both block cases should count as missed: %+v", g.Checks)
	}
}

// writeJSON saves v to a temp file and returns its path.
func writeJSON(t *testing.T, v any) string {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	p := t.TempDir() + "/r.json"
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCompareFindsFixedBrokenAndRegression(t *testing.T) {
	res := func(id, exp, got string, ok bool) Result {
		return Result{ID: id, Expected: exp, Got: got, GotTier: tierOf(got), Correct: ok, TierCorrect: ok}
	}
	old := Score([]Result{
		res("a", "M0", "M0", true),
		res("b", "M3.c", "M0", false),  // will be fixed
		res("c", "M3.a", "M3.a", true), // will break
		res("gone", "M0", "M0", true),
	})
	cur := Score([]Result{
		res("a", "M0", "M0", true),
		res("b", "M3.c", "M3.c", true),
		res("c", "M3.a", "M2.b", false),
		res("new", "M0", "M0", true),
	})
	o, err := LoadSnapshot(writeJSON(t, old))
	if err != nil {
		t.Fatal(err)
	}
	n, err := LoadSnapshot(writeJSON(t, cur))
	if err != nil {
		t.Fatal(err)
	}
	c := Compare(o, n)
	if len(c.Fixed) != 1 || c.Fixed[0].ID != "b" {
		t.Errorf("Fixed = %+v, want b", c.Fixed)
	}
	if len(c.Broken) != 1 || c.Broken[0].ID != "c" || c.Broken[0].After != "M2.b" {
		t.Errorf("Broken = %+v, want c -> M2.b", c.Broken)
	}
	if fmt.Sprint(c.OnlyOld, c.OnlyNew) != "[gone] [new]" {
		t.Errorf("only-in-one = %v %v", c.OnlyOld, c.OnlyNew)
	}
	if !c.Regressed() {
		t.Error("a broken case must count as a regression")
	}
	var out bytes.Buffer
	c.Print(&out)
	for _, want := range []string{"Fixed (wrong -> right) (1)", "Broken (right -> wrong) (1)", "Result: REGRESSION"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("compare output missing %q:\n%s", want, out.String())
		}
	}

	// Comparing a report with itself is not a regression.
	if Compare(n, n).Regressed() {
		t.Error("identical reports must not regress")
	}
}

func TestCompareReadsMultiRunReports(t *testing.T) {
	run := func(gotB string, okB bool) Report {
		return Score([]Result{
			{ID: "a", Expected: "M0", Got: "M0", GotTier: "benign", Correct: true, TierCorrect: true},
			{ID: "b", Expected: "M3.c", Got: gotB, GotTier: tierOf(gotB), Correct: okB, TierCorrect: okB},
		})
	}
	// b is right in 2 of 3 runs, so it counts as correct overall.
	m := Summarise([]Report{run("M3.c", true), run("M0", false), run("M3.c", true)})
	s, err := LoadSnapshot(writeJSON(t, m))
	if err != nil {
		t.Fatal(err)
	}
	if s.Runs != 3 || !s.Cases["b"].Correct || s.Cases["b"].Answer != "M3.c/M0/M3.c" {
		t.Errorf("multi-run snapshot = %+v", s)
	}
}

func TestUsageMeterCountsTokensAndKeepsResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"M3.c"}}],
			"usage":{"prompt_tokens":1400,"completion_tokens":9,
			"prompt_tokens_details":{"cached_tokens":1000},
			"completion_tokens_details":{"reasoning_tokens":4}}}`))
	}))
	defer srv.Close()

	meter := &usageMeter{base: http.DefaultTransport}
	saved := http.DefaultTransport
	http.DefaultTransport = meter
	defer func() { http.DefaultTransport = saved }()

	judge := engine.NewHTTPClient(srv.URL, "k", "m", nil, nil)
	res := Run(context.Background(), judge, []Case{
		{ID: "a", Kind: "llm", Reasoning: "x", Expected: "M3.c"},
		{ID: "b", Kind: "llm", Reasoning: "y", Expected: "M3.c"},
	})
	if !res[0].Correct || !res[1].Correct {
		t.Fatalf("the judge must still parse responses through the meter: %+v", res)
	}
	u := meter.Snapshot()
	if u.Calls != 2 || u.InputTokens != 2800 || u.CachedTokens != 2000 || u.OutputTokens != 18 || u.ReasoningTokens != 8 {
		t.Errorf("usage = %+v", u)
	}

	// 800 fresh input at $2.40/M + 2000 cached at $0.12/M + 18 output at $12/M.
	cost := u.WithCost(Prices{Input: 2.40, CachedInput: 0.12, Output: 12}).CostUSD
	want := (800*2.40 + 2000*0.12 + 18*12) / 1e6
	if cost < want-1e-12 || cost > want+1e-12 {
		t.Errorf("cost = %v, want %v", cost, want)
	}
	if u.WithCost(Prices{}).CostUSD != 0 {
		t.Error("no prices should mean no cost")
	}
	if got := u.Sub(Usage{Calls: 1, InputTokens: 1400}); got.Calls != 1 || got.InputTokens != 1400 {
		t.Errorf("Sub = %+v", got)
	}
}

func TestCleanEndpointDropsSecrets(t *testing.T) {
	got := cleanEndpoint("https://user:secret@example.com/openai/v1/chat/completions?api-key=abc123")
	if got != "https://example.com/openai/v1/chat/completions" {
		t.Errorf("cleanEndpoint = %q", got)
	}
	if strings.Contains(got, "secret") || strings.Contains(got, "abc123") {
		t.Error("endpoint must not keep credentials")
	}
}

func TestMetaRecordsSettingsAndPrices(t *testing.T) {
	t.Setenv("ADRIAN_LLM_OMIT_SAMPLING_PARAMS", "TRUE")
	if !envTrue("ADRIAN_LLM_OMIT_SAMPLING_PARAMS") {
		t.Error("TRUE should read as true")
	}
	t.Setenv("ADRIAN_LLM_OMIT_SAMPLING_PARAMS", "no")
	if envTrue("ADRIAN_LLM_OMIT_SAMPLING_PARAMS") {
		t.Error("no should read as false")
	}

	p := Prices{Input: 2.4, CachedInput: 0.12, Output: 12}
	m := Meta{Backoff: "2s", OmitSamplingParams: true, Prices: &p}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"backoff":"2s"`, `"omit_sampling_params":true`, `"prices_usd_per_million":{"input":2.4,"cached_input":0.12,"output":12}`} {
		if !strings.Contains(string(data), want) {
			t.Errorf("meta JSON missing %s: %s", want, data)
		}
	}
	if (Prices{}).Known() {
		t.Error("zero prices should be unknown")
	}
}

func TestAutoSavePathIsUniqueAndSafe(t *testing.T) {
	dir := t.TempDir() + "/results"
	if err := ensureResultsDir(dir); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(dir + "/.gitignore"); err != nil || !strings.Contains(string(data), "*") {
		t.Fatalf("results folder should ignore itself in git: %q %v", data, err)
	}

	now := time.Date(2026, 10, 8, 16, 40, 29, 0, time.UTC)
	first := autoSavePath(dir, "gpt-6.1-sol", 1, now)
	if !strings.HasSuffix(first, "/2026-10-08T16-40-29Z_gpt-6.1-sol.json") {
		t.Errorf("first = %s", first)
	}
	os.WriteFile(first, []byte("{}"), 0o644)

	second := autoSavePath(dir, "gpt-6.1-sol", 1, now) // same second: must not overwrite
	if second == first || !strings.HasSuffix(second, "_gpt-6.1-sol-2.json") {
		t.Errorf("second = %s, want a new -2 name", second)
	}

	multi := autoSavePath(dir, "org/model name:v2", 3, now)
	if !strings.HasSuffix(multi, "_org-model-name-v2_runs3.json") {
		t.Errorf("unsafe characters and runs suffix: %s", multi)
	}
}
