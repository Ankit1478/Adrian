// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SecureAgentics

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// Scores are the headline numbers of a report. For a multi-run report
// they are averages.
type Scores struct {
	Total, Exact, Tier, Errors float64
	FalsePositives, Benign     float64
	MissedBlocks, BlockCases   float64
}

// caseStatus is one case's outcome in a report. For a multi-run report a
// case is correct when it was correct in more than half of the runs.
type caseStatus struct {
	Expected string
	Correct  bool
	Answer   string
}

// Snapshot is a saved report reduced to what a comparison needs.
type Snapshot struct {
	Path   string
	Runs   int
	Scores Scores
	Cases  map[string]caseStatus
	Order  []string
}

func scoresOf(r Report) Scores {
	return Scores{
		Total: float64(r.Total), Exact: float64(r.Correct), Tier: float64(r.TierCorrect), Errors: float64(r.Errors),
		FalsePositives: float64(r.FalsePositives), Benign: float64(r.Benign),
		MissedBlocks: float64(r.MissedBlocks), BlockCases: float64(r.Violations),
	}
}

// LoadSnapshot reads a report written with -out, single or multi-run.
func LoadSnapshot(path string) (Snapshot, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Snapshot{}, err
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(data, &probe); err != nil {
		return Snapshot{}, fmt.Errorf("%s: not a report: %w", path, err)
	}
	s := Snapshot{Path: path, Cases: map[string]caseStatus{}}

	if _, multi := probe["runs"]; multi {
		var m MultiReport
		if err := json.Unmarshal(data, &m); err != nil {
			return Snapshot{}, fmt.Errorf("%s: %w", path, err)
		}
		if len(m.Runs) == 0 {
			return Snapshot{}, fmt.Errorf("%s: report has no runs", path)
		}
		s.Runs = len(m.Runs)
		s.Scores = Scores{
			Total: float64(m.Runs[0].Total), Exact: m.AvgCorrect, Tier: m.AvgTierCorrect, Errors: m.AvgErrors,
			FalsePositives: m.AvgFalsePositives, Benign: m.AvgBenign,
			MissedBlocks: m.AvgMissedBlocks, BlockCases: m.AvgViolations,
		}
		for i, first := range m.Runs[0].Results {
			correct := 0
			var answers []string
			for _, r := range m.Runs {
				res := r.Results[i]
				if res.Correct {
					correct++
				}
				answers = append(answers, answerOf(res))
			}
			s.Order = append(s.Order, first.ID)
			s.Cases[first.ID] = caseStatus{Expected: first.Expected, Correct: correct*2 > len(m.Runs), Answer: joinAnswers(answers)}
		}
		return s, nil
	}

	var r Report
	if err := json.Unmarshal(data, &r); err != nil {
		return Snapshot{}, fmt.Errorf("%s: %w", path, err)
	}
	s.Runs = 1
	s.Scores = scoresOf(r)
	for _, res := range r.Results {
		s.Order = append(s.Order, res.ID)
		s.Cases[res.ID] = caseStatus{Expected: res.Expected, Correct: res.Correct, Answer: answerOf(res)}
	}
	return s, nil
}

func answerOf(r Result) string {
	if r.Error != "" {
		return "ERROR"
	}
	return r.Got
}

// joinAnswers shows one answer when all runs agree, otherwise all of them.
func joinAnswers(a []string) string {
	for _, x := range a {
		if x != a[0] {
			return strings.Join(a, "/")
		}
	}
	return a[0]
}

// CaseChange is one case whose outcome differs between the two reports.
type CaseChange struct {
	ID, Expected, Before, After string
}

// Comparison is the difference between an old and a new report.
type Comparison struct {
	Old, New         Snapshot
	Fixed, Broken    []CaseChange
	OnlyOld, OnlyNew []string
}

// Compare matches cases by id. Cases present in only one report are
// listed but not scored, since the datasets differ there.
func Compare(old, new Snapshot) Comparison {
	c := Comparison{Old: old, New: new}
	for _, id := range new.Order {
		n := new.Cases[id]
		o, ok := old.Cases[id]
		if !ok {
			c.OnlyNew = append(c.OnlyNew, id)
			continue
		}
		ch := CaseChange{ID: id, Expected: n.Expected, Before: o.Answer, After: n.Answer}
		switch {
		case !o.Correct && n.Correct:
			c.Fixed = append(c.Fixed, ch)
		case o.Correct && !n.Correct:
			c.Broken = append(c.Broken, ch)
		}
	}
	for _, id := range old.Order {
		if _, ok := new.Cases[id]; !ok {
			c.OnlyOld = append(c.OnlyOld, id)
		}
	}
	sort.Strings(c.OnlyOld)
	sort.Strings(c.OnlyNew)
	return c
}

// Regressed reports whether the new report is worse: a case broke, or
// missed blocks or false positives went up.
func (c Comparison) Regressed() bool {
	return len(c.Broken) > 0 ||
		rate(c.New.Scores.MissedBlocks, c.New.Scores.BlockCases) > rate(c.Old.Scores.MissedBlocks, c.Old.Scores.BlockCases) ||
		rate(c.New.Scores.FalsePositives, c.New.Scores.Benign) > rate(c.Old.Scores.FalsePositives, c.Old.Scores.Benign)
}

func rate(n, d float64) float64 {
	if d == 0 {
		return 0
	}
	return n / d
}

// Print writes the score table and the changed cases.
func (c Comparison) Print(w io.Writer) {
	fmt.Fprintf(w, "Old: %s (%d run(s))\nNew: %s (%d run(s))\n\n", c.Old.Path, c.Old.Runs, c.New.Path, c.New.Runs)
	row := func(name string, on, od, nn, nd float64, higherIsBetter bool) {
		op, np := 100*rate(on, od), 100*rate(nn, nd)
		delta := np - op
		mark := ""
		switch {
		case delta > 0.05 && higherIsBetter, delta < -0.05 && !higherIsBetter:
			mark = "  better"
		case delta < -0.05 && higherIsBetter, delta > 0.05 && !higherIsBetter:
			mark = "  WORSE"
		}
		fmt.Fprintf(w, "  %-16s %6.1f%%  ->  %6.1f%%   (%+.1f)%s\n", name, op, np, delta, mark)
	}
	o, n := c.Old.Scores, c.New.Scores
	row("exact accuracy", o.Exact, o.Total, n.Exact, n.Total, true)
	row("tier accuracy", o.Tier, o.Total, n.Tier, n.Total, true)
	row("false positives", o.FalsePositives, o.Benign, n.FalsePositives, n.Benign, false)
	row("missed blocks", o.MissedBlocks, o.BlockCases, n.MissedBlocks, n.BlockCases, false)
	row("errors", o.Errors, o.Total, n.Errors, n.Total, false)

	list := func(title string, changes []CaseChange) {
		fmt.Fprintf(w, "\n%s (%d)\n", title, len(changes))
		for _, ch := range changes {
			fmt.Fprintf(w, "  %-16s expected %-5s %s -> %s\n", ch.ID, ch.Expected, ch.Before, ch.After)
		}
	}
	list("Fixed (wrong -> right)", c.Fixed)
	list("Broken (right -> wrong)", c.Broken)
	if len(c.OnlyOld)+len(c.OnlyNew) > 0 {
		fmt.Fprintf(w, "\nCases in only one report (not compared): old-only %v, new-only %v\n", c.OnlyOld, c.OnlyNew)
	}
	if c.Regressed() {
		fmt.Fprintln(w, "\nResult: REGRESSION")
	} else {
		fmt.Fprintln(w, "\nResult: no regression")
	}
}
