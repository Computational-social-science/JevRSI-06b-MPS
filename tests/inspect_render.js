// Structural inspection of the SHIPPED renderer against the populated demo.
// Not a test — a review aid. The browser sandbox in this environment blocks both
// file:// and loopback, so the page cannot be opened in a real browser here; the
// renderer is therefore exercised in a vm and its output inspected as markup.
// This is the honest substitute, and it is what the assertions in render_test.js
// are built on.
const fs = require("fs");

const page = fs.readFileSync(process.argv[2] || "state/dashboard_demo.html", "utf8");
const style = page.split("<style>")[1].split("</style>")[0];
let js = page.split("<script>")[1].split("<\/script>")[0];
const cut = js.indexOf("async function tick");
if (cut > 0) js = js.slice(0, cut);

// Run the page's script in the vm context itself rather than as a Function body.
// The page declares `const S`, so passing S in as a parameter collides with it —
// the same redeclaration trap render_test.js already hit once.
// The sink must be ONE object, not a fresh literal per call: `render` writes
// through `$("#app").innerHTML = ...`, and a shim that allocates on every
// querySelector throws that write away, so the page renders into nothing and
// every count below reads zero.
const sink = {innerHTML: ""};
const ctx = {
  console,
  document: {querySelector: () => sink},
  setInterval() {},
  fetch: () => Promise.reject(),
};
const vm = require("vm");
vm.createContext(ctx);
vm.runInContext(js, ctx);
const render = vm.runInContext("render", ctx);

const m = page.match(/const SNAPSHOT = (\{[\s\S]*?\});\n/);
if (!m) { console.error("no SNAPSHOT"); process.exit(1); }
const D = JSON.parse(m[1]);
render(D);
const H = sink.innerHTML;

const count = (re) => (H.match(re) || []).length;
const txt = H.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();

console.log(`  source        ${(fs.statSync(process.argv[2]).size / 1024).toFixed(0)} KB`);
console.log(`  readout cells ${count(/class="cell"/g)}`);
console.log(`  panels        ${count(/class="panel"/g)}`);
console.log(`  table rows    ${count(/<tr/g)}`);
console.log(`  pills         ${count(/class="pill /g)}`);
console.log(`  fig circles   ${count(/<circle/g)}`);
console.log(`  fig gridlines ${count(/class="grid"/g)}`);
console.log(`  legend keys   ${count(/<i style/g)}`);
console.log(`  log lines     ${count(/class="tag-/g)}`);
console.log(`  svg text els  ${count(/<text/g)}`);
console.log(`  raw hex in JS ${count(/#[0-9a-fA-F]{6}/g)}  (0 = colour comes only from tokens)`);
console.log("");
console.log("  readout strip:");
console.log("    " + (H.match(/<div class="cells">[\s\S]*?<\/div>\s*<\/div>\s*<\/div>/) || [""])[0]
  .replace(/<[^>]+>/g, "|").replace(/\|+/g, " | ").replace(/\s*\|\s*/g, " · ").trim().slice(0, 420));
console.log("");
console.log("  bar line drawn   :", /bar \+0\.0231/.test(H));
console.log("  champion line    :", /polyline/.test(H));
console.log("  expected markers :", (H.match(/expected \+/g) || []).length);
console.log("  held-out consumed:", /spent/.test(H), "· majority shown:", /0\.5450|0\.545/.test(H));
console.log("  seed progress    :", (H.match(/\d\/3/g) || []).length, "occurrences");
console.log("  underpowered     :", (H.match(/underpowered/g) || []).length);
console.log("  not-checked flag :", /not checked/.test(H));
console.log("  unescaped tags   :", /<script/i.test(txt) ? "YES (BUG)" : "none");
