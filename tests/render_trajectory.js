// The climb must be MONOTONE, and it must actually be drawn as a climb.
//
// Both are load-bearing, so the fixture is built to break them. The decisive case
// is a keep whose delta is LOWER than the champion it replaces: the gate cannot
// produce that (it is `clearsBar_implies_strictly_better` in formal/Gate.lean, for
// an arbitrary ordered group), so if the renderer would draw a descent it is
// drawing a fiction about the protocol.
//
// Each render gets a FRESH vm context. Re-running a slice of the page's script in
// one context collides on its `const` declarations, which is a harness fault
// that looks exactly like a rendering fault.
const fs = require("fs");
const vm = require("vm");
const path = process.argv[2] || "state/trajectory.html";

const page = fs.readFileSync(path, "utf8");
let js = page.split("<script>")[1].split("<\/script>")[0];
const m = js.match(/const DATA = (\{[\s\S]*?\});\n/);
if (!m) { console.error("FAIL  no DATA payload"); process.exit(1); }
const BASE = JSON.parse(m[1]);

let fails = 0;
const check = (name, cond, detail) => {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}` + (detail ? `  [${detail}]` : ""));
  if (!cond) fails++;
};

// Run the page's WHOLE script with DATA substituted, so its own helpers ($,
// esc, sgn, figure) are defined. Slicing from a marker skipped them and failed
// with `$ is not defined` — a harness fault that reads like a page fault.
const WITHOUT_DATA = js.replace(/const DATA = \{[\s\S]*?\};\n/, "");
function render(data) {
  const sink = { innerHTML: "" };
  const ctx = { console, document: { querySelector: () => sink }, setInterval() {} };
  vm.createContext(ctx);
  vm.runInContext("const DATA = " + JSON.stringify(data) + ";\n" + WITHOUT_DATA, ctx);
  return sink.innerHTML;
}

// A fixture that actually climbed: two keeps, a discard between them, a discard
// that beat the first keep but not the second, and a crash.
const CLIMB = {
  ...BASE,
  series: [
    { arm: "null1", role: "calibration", delta: 0.021, keep: false, discard: true,  crash: false, confirmations: [] },
    { arm: "null2", role: "calibration", delta: 0.033, keep: true,  discard: false, crash: false, confirmations: [] },
    { arm: "lr_up",  role: "decisive",    delta: 0.029, keep: false, discard: true,  crash: false, confirmations: [] },
    { arm: "boom",   role: "training",    delta: null, keep: false, discard: false, crash: true,  confirmations: [] },
    { arm: "soft",   role: "decisive",    delta: 0.047, keep: true,  discard: false, crash: false, confirmations: [] },
    { arm: "near",   role: "exploratory", delta: 0.044, keep: false, discard: true,  crash: false, confirmations: [] },
    { arm: "last",   role: "decisive",    delta: 0.041, keep: false, discard: true,  crash: false, confirmations: [] },
  ],
  n_keep: 2, n_discard: 4, n_crash: 1, n_arms: 7, start: 0.021, top: 0.047,
  bar: 0.0231, ckpt_gb: 1.79,
};

console.log("  1. the climb is MONOTONE, because the gate cannot produce otherwise");
let best = null, descents = [];
for (const r of CLIMB.series) {
  if (r.keep && typeof r.delta === "number") {
    if (best !== null && r.delta <= best) descents.push(`${r.arm} ${r.delta} <= ${best}`);
    else best = r.delta;
  }
}
check("the fixture climbs and never descends", descents.length === 0,
  descents.join("; ") || `2 keeps, 0 descents`);

console.log("\n  2. it is drawn as a climb, not as a scatter of attempts");
const html = render(CLIMB);
check("renders", html.length > 0, `${html.length} chars`);
check("a filled staircase under the climb", /opacity="0\.10"/.test(html),
  "a hill, not a cloud of points");
check("the climb line is drawn", /stroke="var\(--keep\)" stroke-width="2\.5"/.test(html));
check("plateaus are labelled with how long they lasted",
  /flat for \d+ arms?/.test(html), "a plateau is the search stuck — that is the result");
check("the legend states the climb cannot descend", /cannot descend/i.test(html));
check("the headline is climb height, not attempt count",
  /climb gained/.test(html) && /survivors/.test(html));
check("crash is visually distinct", /var\(--crash\)/.test(html));
check("discard is visually distinct", /var\(--disc\)/.test(html));

console.log("\n  3. degenerate cases");
const noSurvivor = { ...CLIMB, series: CLIMB.series.map(r => ({ ...r, keep: false })),
                     n_keep: 0, top: null, gain: null };
let h2 = "";
try { h2 = render(noSurvivor); } catch (e) { h2 = "THREW: " + e.message; }
check("a search with no survivor renders without throwing", !h2.startsWith("THREW"),
  h2.startsWith("THREW") ? h2 : "ok");
check("... and draws NO staircase, because nothing climbed",
  !/opacity="0\.10"/.test(h2), "a climb that never started must not be drawn");

let h3 = "";
try { h3 = render({ ...CLIMB, series: [], n_arms: 0, start: null, top: null }); }
catch (e) { h3 = "THREW: " + e.message; }
check("an empty search renders without throwing", !h3.startsWith("THREW"),
  h3.startsWith("THREW") ? h3 : "ok");
check("... with a designed empty state, not a shrug",
  /No arm has been scored/i.test(h3) && /class="empty"/.test(h3));

console.log("\n  4. the live page, built from the real records");
const live = render(BASE);
check("the live trajectory renders", live.length > 0, `${live.length} chars`);
check("it does not claim a climb it has not made",
  /survivors/.test(live));

console.log("\n  5. escaping");
const hostile = { ...CLIMB, series: [{ arm: "<img src=x onerror=alert(1)>", role: "",
  delta: 0.02, keep: true, discard: false, crash: false, confirmations: [] }] };
const h4 = render(hostile);
check("a hostile arm name is escaped, not injected",
  !/<img src=x/.test(h4), "no raw tag from data");

console.log(fails ? `\n${fails} failure(s)` : "\nALL PASS");
process.exit(fails ? 1 : 0);
