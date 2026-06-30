{{- define "deepseek-v4-flash-pd.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- .Release.Name | trunc 28 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.labels" -}}
helm.sh/chart: {{ include "deepseek-v4-flash-pd.chart" . }}
{{ include "deepseek-v4-flash-pd.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "deepseek-v4-flash-pd.selectorLabels" -}}
app.kubernetes.io/name: {{ include "deepseek-v4-flash-pd.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "deepseek-v4-flash-pd.configMapName" -}}
{{- printf "%s-config" (include "deepseek-v4-flash-pd.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.routerName" -}}
{{- printf "%s-router" (include "deepseek-v4-flash-pd.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.routerImage" -}}
{{- if .Values.router.image -}}
{{- .Values.router.image -}}
{{- else -}}
{{- .Values.global.image -}}
{{- end -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.kvTransferConfigJson" -}}
{{- toJson .Values.vllm.kvTransfer -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.kvTransferConfigJsonForRole" -}}
{{- $cfg := deepCopy .root.Values.vllm.kvTransfer -}}
{{- $_ := set $cfg "kv_role" .role -}}
{{- toJson $cfg -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.modelPath" -}}
{{- printf "%s/%s" .Values.model.basePath .Values.model.name -}}
{{- end -}}

{{- define "deepseek-v4-flash-pd.prepareModel" -}}
set -euo pipefail
base_path={{ .Values.onion.dir | quote }}
model_name={{ .Values.onion.model | quote }}
model_path={{ include "deepseek-v4-flash-pd.modelPath" . | quote }}

validate_model_dir() {
  test -f "$model_path/config.json"
  test -f "$model_path/tokenizer.json" || test -f "$model_path/tokenizer.model"
  test -f "$model_path/model.safetensors.index.json" || find "$model_path" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .
}

if validate_model_dir; then
  echo "Model directory $model_path is already complete, skip Onion download."
  exit 0
fi

if ! command -v oniond >/dev/null 2>&1; then
  echo "oniond is missing from the image" >&2
  exit 127
fi

mkdir -p "$base_path"
oniond download model "$model_name" --turbo --dir "$base_path"

if validate_model_dir; then
  echo "Model directory $model_path is ready after Onion download."
  exit 0
fi

echo "Model files not found under expected path $model_path after Onion download" >&2
exit 1
{{- end -}}
