"""Deployment: website with dependent dropdowns + JSON API.
Run:  python app.py   then open http://127.0.0.1:5000"""
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
predictor = PredictPipeline()                            # loads the saved preprocessor and model once

catalog = pd.read_csv("artifacts/phone_catalog.csv")     # every modern phone, with or without a price
brand_counts = catalog["brand"].value_counts()
BRANDS = sorted(brand_counts[brand_counts >= 3].index)
ALL_CHIPSETS = sorted(catalog["chipset_maker"].dropna().unique())

SPEC_FIELDS = ["ram_gb", "storage_gb", "battery_mah", "screen_inches", "refresh_rate_hz", "charging_watts",
               "main_camera_mp", "weight_grams", "process_nm", "release_year", "has_5g", "has_oled_screen"]
FIELDS = ["brand", "chipset_maker"] + SPEC_FIELDS


def clean_value(value):
    """Turns empty numbers into None so they can be sent as JSON."""
    if pd.isna(value):
        return None
    return value if isinstance(value, str) else float(value)


PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Phone price estimator</title>
<style>
    body { margin: 0; padding: 25px 15px; font-family: Arial, sans-serif; background: #eef2f7; }
    .layout { max-width: 1150px; margin: 0 auto; display: grid;
              grid-template-columns: 1fr 300px; gap: 24px; align-items: start; }   /* left = inputs, right = result */
    .card { background: white; padding: 25px 30px; border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.12); text-align: center; }
    h2 { margin-top: 0; color: #1f3a5f; }
    .hint { font-size: 13px; color: #777; margin-bottom: 18px; }
    .form-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px 20px; }   /* 3 columns of inputs */
    .field { display: flex; flex-direction: column; align-items: center; }
    label { font-size: 14px; color: #444; margin-bottom: 5px; }
    input, select { width: 100%; box-sizing: border-box; padding: 9px; font-size: 15px; text-align: center;
                    border: 1px solid #b8c2cf; border-radius: 6px; }
    input:focus, select:focus { outline: none; border-color: #2f6fed; }
    button { grid-column: 1 / -1; justify-self: center; width: 40%; padding: 11px; font-size: 16px; color: white;
             background: #2f6fed; border: none; border-radius: 6px; cursor: pointer; margin-top: 6px; }
    button:hover { background: #1f55c4; }
    .side { position: sticky; top: 20px; }            /* stays at the top right while you scroll */
    .result { padding: 15px; background: #e8f6ee; border-radius: 8px; }
    .result h3 { margin: 0 0 8px 0; color: #17653a; }
    .result p { margin: 4px 0; color: #333; }
    .placeholder { color: #888; font-size: 14px; }
    @media (max-width: 950px) { .layout { grid-template-columns: 1fr; } .side { position: static; order: -1; } }
    @media (max-width: 650px) { .form-grid { grid-template-columns: 1fr 1fr; } button { width: 80%; } }
</style>
</head>
<body>
<div class="layout">
    <div class="card">
        <h2>Phone price estimator (₹)</h2>
        <div class="hint">Choose a brand, then a model. The specs fill in by themselves and you can still change them.</div>
        <form class="form-grid" onsubmit="estimate(event)">
            <div class="field"><label>Brand</label>
                <select id="brand" onchange="loadModels()">{% for b in brands %}<option>{{ b }}</option>{% endfor %}</select></div>
            <div class="field"><label>Model (this brand only)</label>
                <select id="model" onchange="loadPhone()"></select></div>
            <div class="field"><label>Chipset maker</label><select id="chipset_maker"></select></div>
            <div class="field"><label>RAM (GB)</label><input id="ram_gb" value="8"></div>
            <div class="field"><label>Storage (GB)</label><input id="storage_gb" value="256"></div>
            <div class="field"><label>Battery (mAh)</label><input id="battery_mah" value="5000"></div>
            <div class="field"><label>Screen (inches)</label><input id="screen_inches" value="6.7"></div>
            <div class="field"><label>Refresh rate (Hz)</label><input id="refresh_rate_hz" value="120"></div>
            <div class="field"><label>Charging (W)</label><input id="charging_watts" value="67"></div>
            <div class="field"><label>Main camera (MP)</label><input id="main_camera_mp" value="50"></div>
            <div class="field"><label>Weight (g)</label><input id="weight_grams" value="190"></div>
            <div class="field"><label>Process (nm)</label><input id="process_nm" value="6"></div>
            <div class="field"><label>Release year</label><input id="release_year" value="2025"></div>
            <div class="field"><label>5G</label>
                <select id="has_5g"><option value="1">Yes</option><option value="0">No</option></select></div>
            <div class="field"><label>OLED screen</label>
                <select id="has_oled_screen"><option value="1">Yes</option><option value="0">No</option></select></div>
            <button type="submit">Estimate price</button>
        </form>
    </div>
    <div class="side">
        <div class="card">
            <h2>Estimate</h2>
            <div class="result" id="result"><div class="placeholder">Your estimated price will appear here.</div></div>
        </div>
    </div>
</div>

<script>
const ALL_CHIPSETS = {{ chipsets | tojson }};
const SPEC_FIELDS = {{ spec_fields | tojson }};

function fillSelect(id, items, firstLabel) {
    const select = document.getElementById(id);
    select.innerHTML = "";
    if (firstLabel) {
        const first = document.createElement("option");
        first.value = ""; first.textContent = firstLabel;
        select.appendChild(first);
    }
    items.forEach(function (item) {
        const option = document.createElement("option");
        option.value = item; option.textContent = item;
        select.appendChild(option);
    });
}

async function loadModels() {
    const brand = document.getElementById("brand").value;
    const response = await fetch("/api/models?brand=" + encodeURIComponent(brand));
    const data = await response.json();
    fillSelect("model", data.models, "-- type my own specs --");              // only this brand's phones
    fillSelect("chipset_maker", data.chipsets.length ? data.chipsets : ALL_CHIPSETS);   // only chipsets this brand uses
}

async function loadPhone() {
    const brand = document.getElementById("brand").value;
    const model = document.getElementById("model").value;
    if (!model) { return; }
    const response = await fetch("/api/phone?brand=" + encodeURIComponent(brand) + "&model=" + encodeURIComponent(model));
    const data = await response.json();
    SPEC_FIELDS.forEach(function (name) {
        document.getElementById(name).value = (data[name] === null) ? "" : data[name];
    });
    if (data.chipset_maker) { document.getElementById("chipset_maker").value = data.chipset_maker; }
}

function rupees(number) { return "₹" + Number(number).toLocaleString("en-IN"); }

async function estimate(event) {
    event.preventDefault();
    const values = { brand: document.getElementById("brand").value,
                     model: document.getElementById("model").value,
                     chipset_maker: document.getElementById("chipset_maker").value };
    SPEC_FIELDS.forEach(function (name) { values[name] = document.getElementById(name).value; });

    const response = await fetch("/api/predict", { method: "POST",
                                                   headers: { "Content-Type": "application/json" },
                                                   body: JSON.stringify(values) });
    const data = await response.json();
    let html = "<h3>Expected price: " + rupees(data.expected_price_inr) + "</h3>" +
               "<p>Likely range: " + rupees(data.likely_lowest_inr) + " to " + rupees(data.likely_highest_inr) + "</p>";
    if (data.listed_price_inr) {
        const difference = Math.round((data.listed_price_inr / data.expected_price_inr - 1) * 100);
        html += "<p>Listed price of this model: <b>" + rupees(data.listed_price_inr) + "</b> (" +
                Math.abs(difference) + "% " + (difference < 0 ? "below" : "above") + " the expected price)</p>";
    } else if (values.model) {
        html += "<p>This model has no listed price in the data, so only the estimate is shown.</p>";
    }
    const box = document.getElementById("result");
    box.innerHTML = html; box.style.display = "block";
}

loadModels();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE, brands=BRANDS, chipsets=ALL_CHIPSETS, spec_fields=SPEC_FIELDS)


@app.route("/api/models")
def api_models():
    """Models and chipsets that belong to the chosen brand only."""
    rows = catalog[catalog["brand"] == request.args.get("brand", "")]
    return jsonify({"models": sorted(rows["model"].unique()),
                    "chipsets": sorted(rows["chipset_maker"].dropna().unique())})


@app.route("/api/phone")
def api_phone():
    """Specs and listed price of one model."""
    rows = catalog[(catalog["brand"] == request.args.get("brand", "")) & (catalog["model"] == request.args.get("model", ""))]
    if rows.empty:
        return jsonify({"error": "phone not found"}), 404
    row = rows.iloc[0]
    return jsonify({name: clean_value(row[name]) for name in SPEC_FIELDS + ["chipset_maker", "price_inr"]})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    values = request.get_json()
    result = predictor.predict(CustomData(**{name: values.get(name) for name in FIELDS}).get_data_as_dataframe())
    rows = catalog[(catalog["brand"] == str(values.get("brand")).strip().title()) & (catalog["model"] == values.get("model"))]
    listed = rows.iloc[0]["price_inr"] if not rows.empty else None
    result["listed_price_inr"] = None if listed is None or pd.isna(listed) else round(float(listed), -2)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)