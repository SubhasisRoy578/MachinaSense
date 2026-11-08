import os

app_path = os.path.join("frontend", "src", "App.tsx")
with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "ModelTrainingPage" in line and "lazy" in line:
        lines.insert(i + 1, "const ExplainabilityDashboard = lazy(() => import('./features/explainability/ExplainabilityDashboard').then(m => ({ default: m.ExplainabilityDashboard })));\n")
        break

for i, line in enumerate(lines):
    if "model-training" in line and "<Route" in line:
        lines.insert(i + 1, '            <Route path="explainability" element={<ExplainabilityDashboard />} />\n')
        break

with open(app_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("App.tsx updated!")

# Update Sidebar.tsx
sidebar_path = os.path.join("frontend", "src", "layouts", "Sidebar.tsx")
with open(sidebar_path, "r", encoding="utf-8") as f:
    sb_lines = f.readlines()

for i, line in enumerate(sb_lines):
    if "Model Training" in line and "to: '/model-training'" in line:
        sb_lines.insert(i + 1, "  { name: 'Explainability & Robustness', to: '/explainability', icon: BrainCircuit },\n")
        break

with open(sidebar_path, "w", encoding="utf-8") as f:
    sb_lines = [l for l in sb_lines]
    f.writelines(sb_lines)
print("Sidebar.tsx updated!")

# Update PredictionsPage.tsx
preds_path = os.path.join("frontend", "src", "features", "predictions", "PredictionsPage.tsx")
with open(preds_path, "r", encoding="utf-8") as f:
    preds_content = f.read()

button_to_find = '<span>Explain Benchmark</span>'
button_addition = """                <button
                  type="button"
                  onClick={() => navigate('/explainability')}
                  className="text-[10px] text-emerald-400 hover:text-emerald-300 font-semibold flex items-center gap-1 border border-emerald-500/30 px-2 py-0.5 rounded bg-emerald-500/10 transition-colors"
                  title="Explore Phase 10 Explainability, Gradient Saliency, Temporal Dynamics & Robustness Audits"
                >
                  <ActivitySquare className="w-3 h-3" />
                  <span>Explainability & Robustness</span>
                </button>"""

if button_to_find in preds_content and "navigate('/explainability')" not in preds_content:
    # insert before </button> of explain benchmark or right after
    idx = preds_content.find(button_to_find)
    end_btn = preds_content.find("</button>", idx) + len("</button>")
    preds_content = preds_content[:end_btn] + "\n" + button_addition + preds_content[end_btn:]
    with open(preds_path, "w", encoding="utf-8") as f:
        f.write(preds_content)
    print("PredictionsPage.tsx updated!")

