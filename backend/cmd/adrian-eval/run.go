// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SecureAgentics

package main

import (
	"context"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/secureagentics/Adrian/backend/internal/engine"
)

// Options control how cases are sent to the judge.
type Options struct {
	// Concurrency is how many cases are judged at once. Below 1 means 1.
	Concurrency int
	// Retries is how many extra attempts a case gets after a network or
	// server error. Answers from the judge are never retried.
	Retries int
	// Backoff is the wait before the first retry; it doubles each time.
	Backoff time.Duration
}

// Run judges every case one at a time with no retries.
func Run(ctx context.Context, judge engine.Classifier, cases []Case) []Result {
	return RunWith(ctx, judge, cases, Options{Concurrency: 1})
}

// RunWith judges every case using opt. Results keep the order of cases.
// A multi-step case is judged step by step, in order and in one
// goroutine, so the judge sees the earlier steps as history; it
// contributes one result per step.
func RunWith(ctx context.Context, judge engine.Classifier, cases []Case, opt Options) []Result {
	workers := opt.Concurrency
	if workers < 1 {
		workers = 1
	}
	perCase := make([][]Result, len(cases))
	slots := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for i, c := range cases {
		wg.Add(1)
		slots <- struct{}{}
		go func(i int, c Case) {
			defer wg.Done()
			defer func() { <-slots }()
			steps := c.Unroll()
			out := make([]Result, 0, len(steps))
			for _, step := range steps {
				out = append(out, runOne(ctx, judge, step, opt))
			}
			perCase[i] = out
		}(i, c)
	}
	wg.Wait()

	results := make([]Result, 0, len(cases))
	for _, rs := range perCase {
		results = append(results, rs...)
	}
	return results
}

// runOne judges one case, retrying network and server errors. The
// recorded latency is that of the final attempt.
func runOne(ctx context.Context, judge engine.Classifier, c Case, opt Options) Result {
	wait := opt.Backoff
	for attempt := 0; ; attempt++ {
		start := time.Now()
		v, err := judge.Classify(ctx, c.ToEvent(), "")
		latency := time.Since(start).Milliseconds()

		if err == nil || attempt >= opt.Retries || !isTransient(err) {
			r := grade(c, v, err)
			r.LatencyMS, r.Retries = latency, attempt
			return r
		}
		select {
		case <-time.After(wait):
		case <-ctx.Done():
			r := grade(c, nil, ctx.Err())
			r.Retries = attempt
			return r
		}
		wait *= 2
	}
}

var serverStatus = regexp.MustCompile(`status (429|5\d\d)\b`)

// isTransient reports whether err looks like a network or server problem
// worth retrying. A judge answer with no M-code is not transient: it is
// real judge behaviour and must be measured, not hidden.
func isTransient(err error) bool {
	msg := err.Error()
	if serverStatus.MatchString(msg) {
		return true
	}
	for _, s := range []string{"Client.Timeout", "context deadline exceeded", "connection reset", "connection refused", "EOF", "read tcp", "TLS handshake timeout"} {
		if strings.Contains(msg, s) {
			return true
		}
	}
	return false
}
