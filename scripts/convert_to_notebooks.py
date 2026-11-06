import json
import os
import uuid

def convert_py_to_notebook(py_file_path, output_notebook_path):
    """Convierte un script Python con formato de notebook (# %%) a un notebook de Jupyter compatible con Snowflake."""
    
    with open(py_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    cells = []
    current_cell = []
    current_cell_type = None
    current_cell_name = None
    current_cell_language = None
    
    def save_current_cell():
        if current_cell and current_cell_type:
            cell_meta = {}
            if current_cell_name:
                cell_meta['name'] = current_cell_name
            if current_cell_type == 'code':
                cell_meta['language'] = current_cell_language if current_cell_language else 'python'
            cells.append({
                'cell_type': current_cell_type,
                'metadata': cell_meta,
                'source': current_cell
            })

    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detectar inicio de celda markdown
        if line.strip() == '# %% [markdown]':
            save_current_cell()
            current_cell = []
            current_cell_type = 'markdown'
            current_cell_name = None
            current_cell_language = None
            i += 1
            continue
            
        # Detectar inicio de celda de código SQL
        elif line.strip() == '# %% [sql]':
            save_current_cell()
            current_cell = []
            current_cell_type = 'code'
            current_cell_name = None
            current_cell_language = 'sql'
            i += 1
            continue
        
        # Detectar inicio de celda de código
        elif line.strip() == '# %%':
            save_current_cell()
            current_cell = []
            current_cell_type = 'code'
            current_cell_name = None
            current_cell_language = 'python'
            i += 1
            continue
        
        # Agregar línea a la celda actual
        if current_cell_type == 'markdown':
            # Remover el '#' inicial de las líneas markdown
            if line.startswith('# '):
                content = line[2:]
                # Preservar indentación: reemplazar espacios iniciales con &nbsp;
                stripped = content.lstrip(' ')
                leading = len(content) - len(stripped)
                if leading:
                    content = '&nbsp;' * leading + stripped
                current_cell.append(content)
                if not current_cell_name and content.startswith(('# ', '## ', '### ', '#### ')):
                    # Limpiamos los hashes para obtener el nombre limpio
                    clean_name = content.lstrip('#').strip()
                    if clean_name:
                        current_cell_name = clean_name
            elif line.strip() in ('', '#'):
                current_cell.append('')
            else:
                current_cell.append(line)
        elif current_cell_type == 'code':
            current_cell.append(line)
        else:
            # Si no hay tipo de celda definido, asumir código
            if not current_cell_type:
                current_cell_type = 'code'
            current_cell.append(line)
        
        i += 1
    
    # Agregar última celda
    save_current_cell()
    
    # Si no se encontraron celdas, crear una celda con todo el contenido
    if not cells:
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'source': lines
        })
    
    # Crear estructura del notebook compatible con Snowflake
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    # Ajustar celdas para compatibilidad con Snowflake
    for cell in notebook['cells']:
        # Agregar ID único (requerido por Snowflake)
        cell['id'] = str(uuid.uuid4())
        
        # Convertir source a string (Snowflake usa strings, no arrays)
        if isinstance(cell['source'], list):
            if cell['cell_type'] == 'markdown':
                # Entre líneas NO vacías consecutivas, usar salto duro (  \n = <br>)
                # para que cada línea del .py se muestre en su propia línea visual.
                # Entre bloques separados por línea vacía, conservar el \n\n de párrafo.
                parts = []
                src = cell['source']
                for i, line in enumerate(src):
                    if line != '' and i + 1 < len(src) and src[i + 1] != '':
                        parts.append(line + '  ')  # trailing spaces = hard break
                    else:
                        parts.append(line)
                cell['source'] = '\n'.join(parts)
            else:
                cell['source'] = '\n'.join(cell['source'])
        
        if cell['cell_type'] == 'code':
            # Metadata específica para celdas de código en Snowflake
            if 'language' not in cell['metadata']:
                cell['metadata']['language'] = 'python'
            cell['execution_count'] = None
            cell['outputs'] = []
        elif cell['cell_type'] == 'markdown':
            # Metadata específica para celdas markdown en Snowflake
            cell['metadata']['codeCollapsed'] = True
    
    # Guardar notebook
    with open(output_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Convertido: {py_file_path} -> {output_notebook_path}")

import sys

if __name__ == '__main__':
    migration_dir = './'
    notebooks_dir = os.path.join(migration_dir, 'notebooks')

    # Crear directorio de notebooks si no existe
    os.makedirs(notebooks_dir, exist_ok=True)

    if len(sys.argv) > 1:
        # Modo por argumento: python scripts/convert_to_notebooks.py src/archivo.py
        py_path = sys.argv[1]
        notebook_name = os.path.basename(py_path).replace('.py', '.ipynb')
        notebook_path = os.path.join(notebooks_dir, notebook_name)

        if os.path.exists(py_path):
            convert_py_to_notebook(py_path, notebook_path)
            print(f"\nConversion completada! Notebook guardado en: {notebook_path}")
        else:
            print(f"WARNING: No encontrado: {py_path}")
    else:
        # Modo por defecto: convierte todos los scripts de src/
        src_dir = os.path.join(migration_dir, 'src')
        scripts = sorted(
            f for f in os.listdir(src_dir)
            if f.endswith('.py') and not f.startswith('__')
        )

        converted = 0
        for script in scripts:
            py_path = os.path.join(src_dir, script)
            notebook_name = script.replace('.py', '.ipynb')
            notebook_path = os.path.join(notebooks_dir, notebook_name)

            if os.path.exists(py_path):
                convert_py_to_notebook(py_path, notebook_path)
                converted += 1
            else:
                print(f"WARNING: No encontrado: {py_path}")

        print(f"\nConversion completada! {converted} notebooks guardados en: {notebooks_dir}")
