"""Generates the Grafana dashboard JSON for the CardioRisk API.

    python scripts/build_dashboard.py   # writes k8s/monitoring/grafana/dashboards/cardiorisk.json
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "k8s/monitoring/grafana/dashboards/cardiorisk.json"
DS = {"type": "prometheus", "uid": "prometheus"}
COLORS = {"healthy": "#2a9d8f", "disease": "#e76f51", "accent": "#264653", "amber": "#e9c46a"}

_next_id = iter(range(1, 100))


def target(expr: str, legend: str = "", ref: str = "A") -> dict:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}


def stat(title, expr, x, y, w=4, h=4, unit="short", decimals=None, thresholds=None,
         color_mode="value", legend="", description="", text_mode="value"):
    steps = thresholds or [{"color": COLORS["accent"], "value": None}]
    return {
        "id": next(_next_id), "type": "stat", "title": title, "description": description,
        "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(expr, legend)],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals,
                                     "thresholds": {"mode": "absolute", "steps": steps},
                                     "color": {"mode": "thresholds"}}, "overrides": []},
        "options": {"colorMode": color_mode, "graphMode": "area", "textMode": text_mode,
                    "reduceOptions": {"calcs": ["lastNotNull"]}},
    }


def timeseries(title, targets, x, y, w=12, h=8, unit="short", stack=False, overrides=None,
               description="", fill=10):
    return {
        "id": next(_next_id), "type": "timeseries", "title": title, "description": description,
        "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": targets,
        "fieldConfig": {
            "defaults": {"unit": unit, "custom": {
                "lineWidth": 2, "fillOpacity": fill, "showPoints": "never",
                "stacking": {"mode": "normal" if stack else "none"}}},
            "overrides": overrides or [],
        },
        "options": {"legend": {"displayMode": "table", "placement": "right",
                               "calcs": ["lastNotNull", "max"]},
                    "tooltip": {"mode": "multi"}},
    }


def color_override(name, color):
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}


def row(title, y):
    return {"id": next(_next_id), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


REQ = 'cardiorisk_http_requests_total{job="cardiorisk-api"}'
REQ_5XX = 'cardiorisk_http_requests_total{job="cardiorisk-api",status=~"5.."}'
REQ_422 = 'cardiorisk_http_requests_total{job="cardiorisk-api",status="422"}'
LAT = 'cardiorisk_http_request_duration_seconds_bucket{job="cardiorisk-api",path="/predict"}'

panels = [
    row("Service health", 0),
    stat("Pods up", 'sum(up{job="cardiorisk-api"})', 0, 1,
         thresholds=[{"color": "red", "value": None}, {"color": COLORS["amber"], "value": 1},
                     {"color": COLORS["healthy"], "value": 2}],
         description="API pods successfully scraped by Prometheus"),
    stat("Requests / s", f'sum(rate({REQ}[1m]))', 4, 1, unit="reqps", decimals=1),
    stat("p95 latency /predict",
         f'histogram_quantile(0.95, sum by (le) (rate({LAT}[5m])))', 8, 1, unit="s", decimals=3,
         thresholds=[{"color": COLORS["healthy"], "value": None},
                     {"color": COLORS["amber"], "value": 0.1}, {"color": "red", "value": 0.25}]),
    stat("5xx error rate",
         f'(sum(rate({REQ_5XX}[5m])) or vector(0)) / clamp_min(sum(rate({REQ}[5m])), 1e-9)',
         12, 1, unit="percentunit", decimals=2,
         thresholds=[{"color": COLORS["healthy"], "value": None}, {"color": "red", "value": 0.02}]),
    stat("Rejected inputs (422)",
         f'(sum(rate({REQ_422}[5m])) or vector(0)) / clamp_min(sum(rate({REQ}[5m])), 1e-9)',
         16, 1, unit="percentunit", decimals=1,
         thresholds=[{"color": COLORS["accent"], "value": None}, {"color": COLORS["amber"], "value": 0.2}]),
    stat("Model version", 'max by (version) (cardiorisk_model_info)', 20, 1, legend="{{version}}",
         color_mode="none", text_mode="name"),

    row("Traffic & latency", 5),
    timeseries("Requests by endpoint and status",
               [target(f'sum by (path, status) (rate({REQ}[1m]))', "{{path}} {{status}}")],
               0, 6, unit="reqps"),
    timeseries("Latency percentiles - /predict",
               [target(f'histogram_quantile({q}, sum by (le) (rate({LAT}[5m])))', f"p{int(q * 100)}", r)
                for q, r in [(0.5, "A"), (0.95, "B"), (0.99, "C")]] +
               [target('histogram_quantile(0.95, sum by (le) '
                       '(rate(cardiorisk_inference_duration_seconds_bucket[5m])))',
                       "p95 model only", "D")],
               12, 6, unit="s", fill=0,
               description="End-to-end request latency vs time spent inside the model"),

    row("Model behaviour", 14),
    timeseries("Predictions by risk band",
               [target('sum by (risk_band) (rate(cardiorisk_predictions_total[1m]))', "{{risk_band}}")],
               0, 15, w=8, unit="reqps", stack=True, fill=40,
               overrides=[color_override("low", COLORS["healthy"]),
                          color_override("moderate", COLORS["amber"]),
                          color_override("high", COLORS["disease"])]),
    timeseries("Positive prediction rate (drift watch)",
               [target('cardiorisk:positive_rate:10m', "live (10m window)"),
                target('vector(0.46)', "training baseline", "B")],
               8, 15, w=8, unit="percentunit", fill=0,
               overrides=[color_override("live (10m window)", COLORS["disease"]),
                          color_override("training baseline", "#999999")],
               description="Share of requests predicted as heart disease vs the 46% seen in training"),
    {
        "id": next(_next_id), "type": "bargauge", "title": "Predicted probability distribution (15m)",
        "datasource": DS, "gridPos": {"x": 16, "y": 15, "w": 8, "h": 8},
        "targets": [{**target('sum by (le) (increase(cardiorisk_prediction_probability_bucket[15m]))',
                              "{{le}}"), "format": "heatmap"}],
        "fieldConfig": {"defaults": {"unit": "short", "color": {"mode": "continuous-GrYlRd"}},
                        "overrides": []},
        "options": {"orientation": "vertical", "displayMode": "gradient",
                    "reduceOptions": {"calcs": ["lastNotNull"]}},
    },

    row("Resources", 23),
    timeseries("CPU per pod",
               [target('sum by (pod) (rate(container_cpu_usage_seconds_total'
                       '{namespace="cardiorisk",container="api"}[1m]))', "{{pod}}")],
               0, 24, w=8, unit="cores"),
    timeseries("Memory per pod",
               [target('sum by (pod) (container_memory_working_set_bytes'
                       '{namespace="cardiorisk",container="api"})', "{{pod}}")],
               8, 24, w=8, unit="bytes"),
    timeseries("In-flight requests per pod",
               [target('cardiorisk_http_requests_in_flight', "{{pod}}")],
               16, 24, w=8, unit="short"),
]

dashboard = {
    "uid": "cardiorisk-api",
    "title": "CardioRisk API",
    "description": "Traffic, latency, errors and prediction behaviour of the heart disease model",
    "tags": ["cardiorisk", "mlops"],
    "timezone": "browser",
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "schemaVersion": 41,
    "version": 1,
    "editable": False,
    "panels": panels,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {OUT} ({len(panels)} panels)")
