"""
AgentHallu-Bench: Three-Layer Runtime Interceptor
Paper: AgentHallu-Bench: Measuring and Mitigating Error Propagation in Tool-Using Language Agents
Author: Bhaskar Gurram, Zasti Inc.

Best configuration from threshold sweep: tau_L1=1, tau_L2=2, L3=on
F1=0.842, Precision=0.923, Recall=0.774, Abstention=65.0%
"""

class AgentHalluInterceptor:
    def __init__(self, tau_L1=1, tau_L2=2, L3_enabled=True):
        self.tau_L1 = tau_L1   # schema validation threshold
        self.tau_L2 = tau_L2   # keyword monitor threshold
        self.L3_enabled = L3_enabled  # output consistency check
        self.UNCERTAINTY_KEYWORDS = [
            "error", "invalid", "unknown", "missing",
            "incorrect", "wrong", "cannot"
        ]

    def check_L1(self, tool_response):
        """Layer 1: Schema validation - checks parameter errors and tool error responses."""
        flags = 0
        if tool_response.get("error"):
            flags += 1
        if tool_response.get("status") in ["error", "failed", "invalid"]:
            flags += 1
        return flags >= self.tau_L1

    def check_L2(self, chain_of_thought):
        """Layer 2: Thought-keyword monitor - scans CoT for uncertainty keywords."""
        hits = sum(1 for kw in self.UNCERTAINTY_KEYWORDS if kw in chain_of_thought.lower())
        return hits >= self.tau_L2

    def check_L3(self, final_answer, tool_response):
        """Layer 3: Output consistency - detects success claims contradicting errors."""
        if not self.L3_enabled:
            return False
        answer_claims_success = any(
            w in final_answer.lower()
            for w in ["successfully", "completed", "done", "confirmed"]
        )
        tool_had_error = tool_response.get("error") or tool_response.get("status") == "error"
        return answer_claims_success and tool_had_error

    def should_abstain(self, tool_response, chain_of_thought, final_answer):
        """
        Main entry point. Returns True if agent should abstain.
        Best config: tau_L1=1, tau_L2=2, L3=on achieves F1=0.842
        """
        if self.check_L1(tool_response):
            return True, "L1_schema_error"
        if self.check_L2(chain_of_thought):
            return True, "L2_keyword_trigger"
        if self.check_L3(final_answer, tool_response):
            return True, "L3_consistency_violation"
        return False, None
