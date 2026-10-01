// Execute the dashboard's own renderer against real data and assert on the
// output. The sandboxed browser can reach neither loopback nor file://, and a
// screenshot would not catch a silently-empty chart anyway -- this does.
//
//   node tests/render_test.js [snapshot.html]
//
// It extracts the renderer from the shipped page, so it tests the SHIPPED code
// rather than a copy, and runs it in a `vm` context whose globals are the two
// things the page touches: `document` (one element, capturing innerHTML) and
// `fetch` (which must not be reached in snapshot mode).
//
// The first version of this harness passed `document`, `$` and `fetch` as
// `new Function` parameters -- and the page declares `$` itself, so that was a
// redeclaration and the harness never ran at all. A test that cannot execute is
// not a test, so the shims are globals now and the page runs unmodified.

const fs = require("fs");
const vm = require("vm");

const file = process.argv[2] || "state/dashboard_snapshot.html";
const page = fs.readFileSync(file, "utf8");
const js = page.split("<script>")[1].split("</script>")[0];
const cut = js.indexOf("async function tick");
const core = (cut > 0 ? js.slice(0, cut) : js) + "\n;globalThis.__R = {render, chart};";

const dataMatch = js.match(/const SNAPSHOT = (\{[\s\S]*?\});\n/);
if (!dataMatch) { console.error("FAIL  no SNAPSHOT payload in the page"); process.exit(1); }
const DATA = JSON.parse(dataMatch[1]);

let captured = "";
const sandbox = {
  console,
  document: { querySelector: () => ({ set innerHTML(v) { captured = v; }, get innerHTML() { return captured; } }) },
  // Snapshot mode must never reach the network. If it does, that is a failure:
  // an offline snapshot that tries to fetch renders nothing.
  fetch: () => { throw new Error("snapshot must not fetch"); },
  setInterval: () => {},
  JSON, Math, Date, Number, String, Array, Object, isFinite, parseInt, parseFloat,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
try {
  vm.runInContext(core, sandbox, { timeout: 5000 });
} catch (e) {
  console.error(`FAIL  renderer threw while loading: ${e.message}`);
  process.exit(1);
}
const { render, chart } = sandbox.__R;
console.log("  PASS  the page's own renderer loads and exports render/chart");

let fails = 0;
const check = (n, c, d) => { console.log(`  ${c ? "PASS" : "FAIL"}  ${n}` + (d ? `  [${d}]` : "")); if (!c) fails++; };
const H = () => captured;

// --- against the REAL data currently on disk
render(DATA);
check("produces markup", H().length > 800, `${H().length} chars`);
check("has the title", H().includes("RSI-Jev"));
check("renders the readout strip", (H().match(/class="cell"/g) || []).length >= 6,
      `${(H().match(/class="cell"/g) || []).length} cells`);
check("the strip carries the scientific readouts, not just counts",
      ["noise floor", "bar", "champion", "held-out"].every(k => H().includes(k)));
check("states the held-out set's state", /held-out/.test(H()) && /never read|spent/.test(H()));
check("has a health table", H().includes("<table>"));
check("reports arm progress", /arms/.test(H()) && /of the agenda/.test(H()));
check("shows the champion or says there is none",
      /champion/.test(H()) && (DATA.champion && !DATA.inconsistent || /none yet/.test(H())));

// --- the chart, with arms, since none have completed yet
const A = (arm, delta, exp, verdict, under) =>
  ({ arm, axis: "training", delta, expected_effect: exp, verdict, underpowered: under,
     change: "x", reason: "r", top1_candidate: 0.5, top1_control: 0.4, ts: 0 });
const withArms = {
  ...DATA,
  power: { bar: 0.02, measured: true, n_null_arms: 4, paired_sd: 0.011,
           mde: { 1: null, 2: 0.019 }, detectable: {} },
  arms: [A("null_floor", 0.0665, 0, "kept", false),
         A("champion_base", 0.1300, 0.13, "kept_pending_confirm", false),
         A("model_mlp_combine", 0.0010, 0.002, "rejected", true),
         A("train_lower_layers_soft", -0.0040, 0.02, "not_confirmed", false)],
  champion: { arm: "champion_base", pooled_top1: 0.5945 },
};
render(withArms);
check("with arms, the chart has an svg", H().includes("<svg"));
check("the bar reference line is drawn", H().includes("bar +"));
check("the champion polyline is drawn", H().includes("polyline"));
check("all four verdicts render in distinct colours",
      ["good", "warn", "quiet", "bad"].every(c => H().includes(`var(--${c})`)));
check("underpowered is labelled on the row", H().includes("underpowered"));
check("deltas are signed, both directions",
      H().includes("+0.1300") && H().includes("-0.0040"));
check("expected effect is shown per arm", H().includes("+0.0200"));
check("the champion is named when there is one", /champion_base/.test(H()));

// --- degenerate inputs must not throw or divide by zero
render({ ...withArms, arms: [] });
check("zero arms renders a DESIGNED empty state, not a shrug",
      H().includes("No arm has been scored") && H().includes("class=\"empty\""));
render({ ...withArms, arms: [withArms.arms[0]] });
check("one arm renders (X scale would divide by zero)", H().includes("<svg"));
render({ ...DATA, power: null, champion: null, pending_confirm: null, arms: [], inconsistent: null });
check("null power / null champion does not throw", H().includes('class="cells"'));
render({ ...withArms, arms: withArms.arms.map(a => ({ ...a, delta: null })) });
check("arms with no delta render", H().includes('class="cells"'));

// --- an inconsistent champion must be surfaced, never rendered as fact
render({ ...withArms, inconsistent: "champion.json names 'null2' but that arm has no record" });
check("an inconsistent champion raises a visible banner",
      /inconsistent state/i.test(H()) && H().includes("verdict bad"));

// --- the diagnosis panel: the four verdicts and the never-ran case
const DOC = (verdict, ok, diag) => ({ verdict, ok, diagnosis: diag, when_h: "12:00:00",
  auditors: { vitals: { n: 7, ok: ok ? 7 : 5, critical: ok ? [] : ["V1 down"],
                        major: [] },
              methodology: { n: 4, ok: ok ? 4 : 3, critical: ok ? [] : ["I5 floor"],
                             major: [] } } });
render({ ...withArms, doctor: DOC("healthy", true, "all auditors agree") });
check("healthy verdict renders", H().includes("healthy") && H().includes("all auditors agree"));
render({ ...withArms, doctor: DOC("wrong", false, "the loop is running but the SCIENCE is not") });
check("WRONG is unmistakable", H().includes("wrong"), "must not be mistakable for degraded");
check("WRONG shows the auditors and their failures",
      H().includes("I5 floor") && H().includes("V1 down"));
render({ ...withArms, doctor: DOC("broken", false, "the loop died") });
check("BROKEN is distinguished from WRONG",
      /broken/.test(H()) && !/running the wrong science/.test(H()));
render({ ...withArms, doctor: null });
check("a missing doctor says the pipeline is NOT being checked",
      /never run/i.test(H()) && /not being diagnosed/i.test(H()),
      "silence about the monitor is the worst case");
render({ ...withArms, doctor: DOC("unreadable", false, "doctor.json could not be parsed") });
check("an unreadable doctor is shown as such, not as healthy",
      H().includes("unreadable"));

// --- a stale verdict must be visible, not silently believed
// The page reads doctor.json, which the watchdog writes on its own interval. An
// unlabelled stale verdict is indistinguishable from a current one.
const stale = (() => { const d = DOC("healthy", true, "fine");
  d.when = Date.now()/1000 - 7200; return d; })();
render({ ...withArms, doctor: stale });
check("a stale diagnosis is labelled as stale", /stale\s*2h/.test(H()),
      "2h old must not read as current");
check("... and says how old it is", /as of/i.test(H()));
const fresh = (() => { const d = DOC("healthy", true, "fine");
  d.when = Date.now()/1000 - 60; return d; })();
render({ ...withArms, doctor: fresh });
check("a current diagnosis is NOT labelled stale", !/stale/.test(H()));

// --- escaping: a hostile arm name must not become markup
render({ ...withArms, arms: [A("<script>alert(1)</script>", 0.1, 0.1, "rejected", false)] });
check("a hostile arm name is escaped, not injected",
      !H().includes("<script>alert(1)"), "no raw script tag from data");

console.log(`\n${fails} failure(s)`);
process.exit(fails ? 1 : 0);
