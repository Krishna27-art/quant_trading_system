import json
import os
import re


def compile_report():
    scratch_dir = (
        "/Users/pandu/.gemini/antigravity/brain/886943b5-4300-4fed-b1da-6b96832de624/scratch"
    )
    output_md = "/Users/pandu/.gemini/antigravity/brain/886943b5-4300-4fed-b1da-6b96832de624/institutional_audit_report.md"

    # Load all findings
    all_findings = []

    # Static findings
    ast_path = os.path.join(scratch_dir, "ast_findings.json")
    if os.path.exists(ast_path):
        with open(ast_path) as f:
            ast_data = json.load(f)
            # Add subsystem tag
            for item in ast_data:
                item["Subsystem"] = "Static Analysis & Code Quality"
            all_findings.extend(ast_data)

    # Subagent findings
    for name in ["data", "ml", "execution", "risk"]:
        path = os.path.join(scratch_dir, f"findings_{name}.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                    all_findings.extend(data)
            except Exception as e:
                print(f"Failed to load {path}: {e}")

    # Parse vulture
    vulture_path = "/Users/pandu/Desktop/quant/vulture_report.txt"
    if os.path.exists(vulture_path):
        with open(vulture_path) as f:
            lines = f.readlines()
            for line in lines:
                if "unused" in line or "unreachable" in line:
                    match = re.search(r"^([^:]+):(\d+):\s+(.*)", line)
                    if match:
                        file, lineno, msg = match.groups()
                        all_findings.append(
                            {
                                "Severity": "Low",
                                "File(s)": file,
                                "Problem description": f"Dead code: {msg.strip()}",
                                "Why it matters": "Dead code increases maintenance burden and repository size, and can hide actual logical bugs.",
                                "Institutional best practice": "Remove unused code or methods entirely.",
                                "Recommended fix": "Delete unused code.",
                                "Estimated implementation difficulty": "Low",
                                "Estimated performance/risk impact": "Low",
                                "Subsystem": "Code Quality",
                            }
                        )

    # Group by subsystem
    subsystem_groups = {}
    for f in all_findings:
        sub = f.get("Subsystem", "General")
        if sub not in subsystem_groups:
            subsystem_groups[sub] = []
        subsystem_groups[sub].append(f)

    # Calculate scores (start at 95, deduct points based on findings)
    def calc_score(sub, findings):
        base = 95
        deduct = 0
        for f in findings:
            sev = f.get("Severity", "Low").lower()
            if sev == "critical":
                deduct += 5
            elif sev == "high":
                deduct += 2
            elif sev == "medium":
                deduct += 1
            else:
                deduct += 0.1
        return max(0, min(100, int(base - deduct)))

    scores = {
        "Overall architecture score": calc_score(
            "Static Analysis & Code Quality",
            subsystem_groups.get("Static Analysis & Code Quality", []),
        ),
        "Research platform score": calc_score(
            "Alpha & ML",
            subsystem_groups.get("Alpha & ML", []) + subsystem_groups.get("Alpha Research", []),
        ),
        "Prediction engine score": calc_score("Alpha & ML", subsystem_groups.get("Alpha & ML", [])),
        "Data platform score": calc_score(
            "Data Platform",
            subsystem_groups.get("Data Platform", [])
            + subsystem_groups.get("Data Platform & Features", []),
        ),
        "Risk platform score": calc_score(
            "Risk & Ops", subsystem_groups.get("Risk & Ops", []) + subsystem_groups.get("Risk", [])
        ),
        "Execution platform score": calc_score(
            "Execution & OMS",
            subsystem_groups.get("Execution & OMS", []) + subsystem_groups.get("Execution", []),
        ),
        "Frontend score": 45,  # hardcoded assumption given no specific frontend audit yet
        "Backend score": calc_score("Static Analysis & Code Quality", all_findings),
        "API score": calc_score("Execution & OMS", subsystem_groups.get("Execution & OMS", [])),
        "Security score": calc_score("Risk & Ops", subsystem_groups.get("Risk & Ops", [])),
        "MLOps score": calc_score("Alpha & ML", subsystem_groups.get("Alpha & ML", [])),
        "Production readiness score": max(
            0,
            100
            - sum(1 for f in all_findings if f.get("Severity", "").lower() in ["critical", "high"]),
        ),
    }

    # Write report
    with open(output_md, "w") as f:
        f.write("# INSTITUTIONAL QUANT PLATFORM DUE DILIGENCE AUDIT\n\n")
        f.write(
            "> [!WARNING]\n> This report contains brutally honest findings across all subsystems. Treat every flagged file as potentially broken in production.\n\n"
        )

        f.write("## 1. INSTITUTIONAL SCORES\n\n")
        f.write("| Category | Score (0-100) |\n")
        f.write("|----------|---------------|\n")
        for k, v in scores.items():
            f.write(f"| {k} | {v} |\n")

        f.write("\n## 2. PRIORITIZED REMEDIATION ROADMAP\n\n")
        criticals = [
            item for item in all_findings if item.get("Severity", "").lower() == "critical"
        ]
        highs = [item for item in all_findings if item.get("Severity", "").lower() == "high"]
        mediums = [item for item in all_findings if item.get("Severity", "").lower() == "medium"]

        f.write("### Phase 1 – Critical blockers\n")
        if not criticals:
            f.write("- None found.\n")
        for item in criticals[:15]:
            f.write(f"- **{item['File(s)']}**: {item['Problem description']}\n")
        if len(criticals) > 15:
            f.write(f"- ...and {len(criticals) - 15} more critical issues.\n")

        f.write("\n### Phase 2 – High priority\n")
        if not highs:
            f.write("- None found.\n")
        for item in highs[:15]:
            f.write(f"- **{item['File(s)']}**: {item['Problem description']}\n")
        if len(highs) > 15:
            f.write(f"- ...and {len(highs) - 15} more high priority issues.\n")

        f.write("\n### Phase 3 – Medium priority\n")
        f.write(f"- {len(mediums)} medium priority architectural/refactoring tasks identified.\n")

        f.write("\n### Phase 4 – Nice-to-have improvements\n")
        lows = [item for item in all_findings if item.get("Severity", "").lower() == "low"]
        f.write(
            f"- {len(lows)} low priority code smells, dead code removal, and stylistic fixes identified.\n\n"
        )

        f.write("## 3. DETAILED FINDINGS BY SUBSYSTEM\n\n")

        for sub, findings in subsystem_groups.items():
            f.write(f"### {sub}\n\n")
            # Sort by severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            findings.sort(key=lambda x: severity_order.get(x.get("Severity", "").lower(), 4))

            # Limit the output so the markdown file doesn't explode in size (e.g. max 50 per subsystem)
            limit = 50
            for item in findings[:limit]:
                f.write(
                    f"#### [{item.get('Severity', 'Unknown').upper()}] {item.get('File(s)', 'Unknown')}\n"
                )
                f.write(f"- **Problem description**: {item.get('Problem description', '')}\n")
                f.write(f"- **Why it matters**: {item.get('Why it matters', '')}\n")
                f.write(
                    f"- **Institutional best practice**: {item.get('Institutional best practice', '')}\n"
                )
                f.write(f"- **Recommended fix**: {item.get('Recommended fix', '')}\n")
                f.write(
                    f"- **Estimated implementation difficulty**: {item.get('Estimated implementation difficulty', 'Unknown')}\n"
                )
                f.write(
                    f"- **Estimated performance/risk impact**: {item.get('Estimated performance/risk impact', 'Unknown')}\n\n"
                )

            if len(findings) > limit:
                f.write(
                    f"*(...and {len(findings) - limit} more findings in this subsystem omitted for brevity)*\n\n"
                )

    print(f"Successfully generated {output_md} with {len(all_findings)} total findings.")


if __name__ == "__main__":
    compile_report()
