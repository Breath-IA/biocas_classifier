# ── Argumentos del script ─────────────────────────────────────────────────
# --data-dir : carpeta raíz donde quedan train/{wav,json} y test/{wav,json}
#              (se reenvía tal cual a data_manifest.py al final)
# --config   : ruta del yaml de pipeline a actualizar (se reenvía también)
DATA_DIR_ARG="./data/raw_data"
DATA_OUT_DIR_ARG="./data/processed"
CONFIG_ARG="config/pipeline.yaml"


# pull dataset
mkdir -p $DATA_DIR_ARG
mkdir -p $DATA_OUT_DIR_ARG
cd ./data
if [ -d "SPRSound" ]; then
    echo "El repositorio ya existe"
else
    git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
    echo "Repo descargado"
fi
cd SPRSound

pwd
##

# Configuración de rutas basada en tu imagen
REPO_DIR="./"  # Ajusta si tu "data" raíz tiene otro nombre
# Si estás ejecutando el script al lado de 'raw_data', usa: REPO_DIR="./raw_data/SPRSound"
USE_SYMLINKS="false"

DATA_DIR="../raw_data"
link_or_copy() {
    local src="$1"
    local dst="$2"

    if [ -e "$dst" ] || [ -L "$dst" ]; then
        return
    fi

    if [ "$USE_SYMLINKS" = "true" ]; then
        ln -s "$(readlink -f "$src")" "$dst"
    else
        cp -p "$src" "$dst"
    fi
}

# Definición de fuentes utilizando tu estructura
SOURCES=(
    "$REPO_DIR/BioCAS2022/train2022_wav|$REPO_DIR/BioCAS2022/train2022_json|train"
    "$REPO_DIR/BioCAS2022/test2022_wav|$REPO_DIR/BioCAS2022/test2022_json|train"
    "$REPO_DIR/BioCAS2023/test2023_wav|$REPO_DIR/BioCAS2023/test2023_json|train"
    "$REPO_DIR/BioCAS2024/test2024_wav|$REPO_DIR/BioCAS2024/test2024_json|train"
    "$REPO_DIR/BioCAS2025/test2025_wav|$REPO_DIR/BioCAS2025/test2025_json|test"
)

declare -A total_entrenamiento_wav=0 total_entrenamiento_json=0
declare -A total_validacion_wav=0 total_validacion_json=0

for source in "${SOURCES[@]}"; do
    IFS='|' read -r wav_src json_src split <<< "$source"

    if [ ! -d "$wav_src" ]; then
        printf " No encontrado: %s – saltando.\n" "$wav_src"
        continue
    fi

    # Extraer el año buscando los 4 dígitos que siguen a 'BioCAS'
    if [[ "$wav_src" =~ BioCAS([0-9]{4}) ]]; then
        year="${BASH_REMATCH[1]}"
    else
        year="unknown"
    fi

    # Crear carpetas de destino de manera dinámica
    wav_dst="$DATA_DIR/$split/wav"
    json_dst="$DATA_DIR/$split/json"
    mkdir -p "$wav_dst" "$json_dst"

    n_wav=0
    n_json=0

    shopt -s nullglob
    for wav_path in "$wav_src"/*.wav; do
        wav_name=$(basename "$wav_path")
        
        # Renombrar usando el año como prefijo
        dst_wav_name="${year}__${wav_name}"
        
        link_or_copy "$wav_path" "$wav_dst/$dst_wav_name"
        ((n_wav++))

        # --- CAMBIO CLAVE: Búsqueda recursiva del archivo .json ---
        target_json_name="${wav_name%.wav}.json"
        
        # find busca de manera descendente dentro de json_src el archivo exacto
        json_path=$(find "$json_src" -type f -name "$target_json_name" -print -quit)
        
        if [ -n "$json_path" ] && [ -f "$json_path" ]; then
            dst_json_name="${year}__${target_json_name}"
            link_or_copy "$json_path" "$json_dst/$dst_json_name"
            ((n_json++))
        fi
    done
    shopt -u nullglob

    # Acumular estadísticas globales
    if [ "$split" = "test" ]; then
        total_entrenamiento_wav=$((total_entrenamiento_wav + n_wav))
        total_entrenamiento_json=$((total_entrenamiento_json + n_json))
    elif [ "$split" = "train" ]; then
        total_validacion_wav=$((total_validacion_wav + n_wav))
        total_validacion_json=$((total_validacion_json + n_json))
    fi

    printf "  %-13s | %-4s → %5d WAVs  %5d JSONs\n" "$split" "$year" "$n_wav" "$n_json"
done
cd ..
rm -rf ./SPRSound
cd ..

python3 data_config.py --data-dir "$DATA_DIR_ARG" --output-dir "$DATA_OUT_DIR_ARG" --config "$CONFIG_ARG"