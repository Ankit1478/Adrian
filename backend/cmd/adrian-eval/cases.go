// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SecureAgentics

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"

	pb "github.com/secureagentics/Adrian/backend/internal/proto"
)

// Case is one labelled test for the judge. It is one JSON object per
// line in the cases file.
//
// kind "llm" describes what the model reasoned and which tools it wants
// to call. kind "tool" describes a tool that actually ran, with its
// input and output.
type Case struct {
	ID                string         `json:"id"`
	Kind              string         `json:"kind"`
	AgentSystemPrompt string         `json:"agent_system_prompt"`
	UserInstruction   string         `json:"user_instruction"`
	Reasoning         string         `json:"reasoning"`
	Response          string         `json:"response"`
	ToolCalls         []CaseToolCall `json:"tool_calls"`
	ToolName          string         `json:"tool_name"`
	Input             string         `json:"input"`
	Output            string         `json:"output"`
	Expected          string         `json:"expected"`
	AlsoOK            []string       `json:"also_ok"`
	Note              string         `json:"note"`
	Tags              []string       `json:"tags"`
}

// CaseToolCall is a tool call the model wants to make (kind "llm").
type CaseToolCall struct {
	Name string `json:"name"`
	Args string `json:"args"`
}

var codePattern = regexp.MustCompile(`^M[0234](\.[a-z])?$`)

func (c Case) validate() error {
	if c.ID == "" {
		return fmt.Errorf("id is required")
	}
	switch c.Kind {
	case "llm":
		if c.Reasoning == "" && c.Response == "" && len(c.ToolCalls) == 0 {
			return fmt.Errorf("kind llm needs reasoning, response or tool_calls")
		}
	case "tool":
		if c.ToolName == "" {
			return fmt.Errorf("kind tool needs tool_name")
		}
	default:
		return fmt.Errorf("kind must be \"llm\" or \"tool\", got %q", c.Kind)
	}
	if !codePattern.MatchString(c.Expected) {
		return fmt.Errorf("expected %q is not a valid M-code (want M0, M2, M2.a, ...)", c.Expected)
	}
	for _, code := range c.AlsoOK {
		if !codePattern.MatchString(code) {
			return fmt.Errorf("also_ok %q is not a valid M-code", code)
		}
	}
	return nil
}

// LoadCases reads JSONL, one case per line. Blank lines are skipped.
// Unknown fields are rejected so a typo such as "expcted" fails loudly
// instead of silently scoring against an empty label.
func LoadCases(r io.Reader) ([]Case, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	var cases []Case
	seen := map[string]bool{}
	for line := 1; sc.Scan(); line++ {
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var c Case
		dec := json.NewDecoder(strings.NewReader(text))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&c); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if err := c.validate(); err != nil {
			return nil, fmt.Errorf("line %d (%s): %w", line, c.ID, err)
		}
		if seen[c.ID] {
			return nil, fmt.Errorf("line %d: duplicate id %q", line, c.ID)
		}
		seen[c.ID] = true
		cases = append(cases, c)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return cases, nil
}

// ToEvent builds the protobuf event the backend would receive, so the
// judge sees exactly what it sees in production.
func (c Case) ToEvent() *pb.PairedEvent {
	ev := &pb.PairedEvent{
		EventId:      c.ID,
		SessionId:    "eval",
		InvocationId: "eval",
		Agent: &pb.AgentContext{
			AgentId:         "eval-agent",
			SystemPrompt:    c.AgentSystemPrompt,
			UserInstruction: c.UserInstruction,
		},
	}
	if c.Kind == "tool" {
		ev.PairType = pb.PairType_PAIR_TYPE_TOOL
		ev.Data = &pb.PairedEvent_Tool{Tool: &pb.ToolPairData{
			ToolName:   c.ToolName,
			ToolCallId: c.ID,
			Input:      c.Input,
			Output:     c.Output,
		}}
		return ev
	}
	llm := &pb.LlmPairData{Output: c.Response, Reasoning: c.Reasoning}
	for _, tc := range c.ToolCalls {
		llm.ToolCalls = append(llm.ToolCalls, &pb.ToolCall{Name: tc.Name, Args: tc.Args})
	}
	ev.PairType = pb.PairType_PAIR_TYPE_LLM
	ev.Data = &pb.PairedEvent_Llm{Llm: llm}
	return ev
}
