"""
package_deploy.py — empaqueta el código necesario para deploy en AWS SageMaker.

El zip resultante contiene todo lo que el dev destinatario necesita para:
    1. python deploy/deploy_sm.py          # empaqueta src/ + config.py → S3
    2. python deploy/pipeline_def.py       # registra los dos pipelines
    3. python deploy/deploy_sm.py --run    # dispara el pipeline

Contenido incluido:
    src/             — scripts del pipeline + features.py
    deploy/          — deploy_sm.py, pipeline_def.py, run_local.py, README.md
    models/          — modelos ya entrenados (tune_meta.json, train_meta.json, lgbm_base.pkl, xgb_base.pkl)
    config.py        — configuración central con SM_* overrides
    requirements.txt

Excluye: data/, reports/, docs/, scripts/, __pycache__, *.pyc, .env, .git/

Uso:
    python scripts/package_deploy.py
    python scripts/package_deploy.py --out dist/
"""

import argparse
import zipfile
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # raíz del proyecto (un nivel arriba de scripts/)

INCLUDE = [
    "src",
    "deploy",
    "models",
    "config.py",
    "requirements.txt",
    ".env.example",
]

EXCLUDE_DIRS  = {"__pycache__", ".git", ".mypy_cache", ".venv", "venv",
                 "node_modules", ".ruff_cache", ".pytest_cache"}
EXCLUDE_EXTS  = {".pyc", ".pyo", ".pyd"}
EXCLUDE_FILES = {".env", ".DS_Store", "Thumbs.db"}


def should_include(path: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return False
    if path.suffix in EXCLUDE_EXTS:
        return False
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    return True


def collect_files(root: Path) -> list[tuple[Path, str]]:
    """Retorna lista de (ruta_absoluta, nombre_en_zip). Estructura preservada dentro de promo-uplift-ml/."""
    entries = []
    for item in INCLUDE:
        p = root / item
        if not p.exists():
            print(f"  omitido (no encontrado): {item}")
            continue
        if p.is_file():
            if should_include(p):
                entries.append((p, f"promo-uplift-ml/{item}"))
        else:
            for f in sorted(p.rglob("*")):
                if f.is_file() and should_include(f):
                    rel = f.relative_to(root)
                    entries.append((f, f"promo-uplift-ml/{rel.as_posix()}"))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empaqueta el código de deploy M1 en un zip listo para enviar."
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT,
        help="Directorio de salida del zip (default: raíz del proyecto)"
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().strftime("%Y%m%d")
    zip_name = f"promo-uplift-ml-deploy-{today}.zip"
    zip_path = out_dir / zip_name

    files = collect_files(ROOT)
    if not files:
        print("No se encontraron archivos para empaquetar.")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for abs_path, arc_name in files:
            zf.write(abs_path, arc_name)

    size_kb = zip_path.stat().st_size / 1024
    print(f"\n  Zip creado: {zip_path}")
    print(f"  Archivos incluidos: {len(files)}   Tamaño: {size_kb:.1f} KB\n")

    # Desglose por directorio
    by_dir: dict[str, list[str]] = {}
    for _, arc in sorted(files):
        parts = arc.split("/")
        top = parts[1] if len(parts) > 2 else "."
        by_dir.setdefault(top, []).append("/".join(parts[1:]))

    for section, items in sorted(by_dir.items()):
        print(f"  {section}/  ({len(items)} archivos)")
        for f in items:
            print(f"    {f}")
    print()

    print("  --- Instrucciones para el dev destinatario ---")
    print(f"  1. Descomprimir: {zip_name}  -> genera carpeta promo-uplift-ml/")
    print("  2. cd promo-uplift-ml")
    print("  3. pip install -r requirements.txt  (requiere Python 3.11+, boto3 ya instalado)")
    print("  4. Configurar AWS:  aws configure  o copiar ~/.aws/credentials")
    print("  5. python deploy/deploy_sm.py          # empaqueta src/ y sube a S3")
    print("  6. python deploy/pipeline_def.py       # registra promo-uplift-training e promo-uplift-scoring")
    print("  7. python deploy/deploy_sm.py --run    # dispara el pipeline deseado")
    print()


if __name__ == "__main__":
    main()
