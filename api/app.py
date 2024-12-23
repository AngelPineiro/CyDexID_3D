from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from .main import interpretar_string_canonico
from .gen_GPUs import mol2pdb
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import base64, io, os, time, sys, uuid, subprocess, tempfile
from .paste_GPUs_06 import GPUPaster
from openbabel import pybel
from openbabel import openbabel as ob
import json
from datetime import datetime
import geoip2.database
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__, static_folder='../public', static_url_path='')
CORS(app)

ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
TEMP_DIR = '/tmp' if ENVIRONMENT == 'production' else 'temp'

def format_log_entry(user_info):
    """Formatea la información del usuario en un formato YAML legible"""
    # Crear una copia del diccionario para no modificar el original
    info = user_info.copy()
    
    # Obtener timestamp o usar el tiempo actual si no existe
    timestamp = info.pop('timestamp') if 'timestamp' in info else datetime.now().isoformat()
    
    # Función helper para formatear valores None
    def format_value(value):
        if value is None:
            return "no disponible"
        return value
    
    yaml_entry = f"""
# ====================================
#  Nueva Actividad de Usuario
# ====================================

Fecha y Hora: {timestamp}

Información del Usuario:
    IP: {info.get('ip_address', 'desconocida')}
    
    Ubicación:
        País: {info.get('country', 'desconocido')}
        Ciudad: {info.get('city', 'desconocida')}
        Coordenadas: 
            Latitud: {info.get('latitude', 'desconocida')}
            Longitud: {info.get('longitude', 'desconocida')}
        Zona Horaria: {info.get('timezone', 'desconocida')}
    
    Información del Navegador:
        User Agent: {info.get('user_agent', 'desconocido')}
        Plataforma: {format_value(info.get('platform'))}
        Idioma: {format_value(info.get('accept_language'))}
    
    Página de Referencia: {format_value(info.get('referer'))}

Actividad:
    Acción: {info.get('action', 'desconocida')}"""

    # Añadir string canónico solo si está presente
    if 'canonical_string' in info:
        yaml_entry += f"""
    String Canónico: {info['canonical_string']}"""

    # Añadir errores si existen
    if 'geoip_error' in info:
        yaml_entry += f"""

Errores:
    Error de Geolocalización: {info['geoip_error']}"""

    yaml_entry += "\n\n# ------------------------------------\n"  # Separador entre entradas
    return yaml_entry

def setup_logging(user_dir):
    """Configura el logging para el directorio específico del usuario"""
    log_file = os.path.join(user_dir, 'user_activity.log')
    
    # Crear un logger único para este directorio
    logger = logging.getLogger(f'user_logger_{user_dir}')
    
    # Si el logger ya tiene handlers, no añadir más
    if not logger.handlers:
        # Configurar el logger
        logger.setLevel(logging.INFO)
        
        # Prevenir la propagación a otros loggers
        logger.propagate = False
        
        class YAMLFormatter(logging.Formatter):
            def format(self, record):
                if isinstance(record.msg, dict):
                    return format_log_entry(record.msg)
                else:
                    return f"""
# ====================================
#  Mensaje del Sistema
# ====================================

Fecha y Hora: {datetime.fromtimestamp(record.created).isoformat()}
Nivel: {record.levelname}
Mensaje: {record.msg}

# ------------------------------------
"""

        # Crear y configurar el handler
        handler = RotatingFileHandler(log_file, maxBytes=1000000, backupCount=3, encoding='utf-8')
        handler.setFormatter(YAMLFormatter())
        logger.addHandler(handler)
    
    return logger

def get_user_info():
    """Recopila información del usuario"""
    # Obtener zona horaria e información del sistema
    import time
    import platform
    
    local_timezone = time.tzname[0]
    
    # Obtener y formatear la información de la plataforma de forma más amigable
    system = platform.system()
    if system == "Darwin":
        system = "macOS"
        # Obtener la versión real de macOS en lugar de la versión del kernel
        mac_ver = platform.mac_ver()[0]  # Ejemplo: "10.15.7"
        system_platform = f"{system} {mac_ver}"
    elif system == "Windows":
        system_platform = f"Windows {platform.release()}"
    elif system == "Linux":
        system_platform = f"Linux {platform.release()}"
    else:
        system_platform = f"{system} {platform.release()}"
    
    user_info = {
        'timestamp': datetime.now().isoformat(),
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent'),
        'accept_language': request.headers.get('Accept-Language'),
        'referer': request.headers.get('Referer'),
        'platform': request.headers.get('Sec-Ch-Ua-Platform') or system_platform,
        'host': request.headers.get('Host'),
        'method': request.method,
        'path': request.path,
        'query_string': request.query_string.decode(),
        'remote_port': request.environ.get('REMOTE_PORT'),
        'request_time': datetime.now().strftime('%H:%M:%S'),
    }
    
    # Intentar obtener información geográfica usando GeoIP2
    try:
        with geoip2.database.Reader('path/to/GeoLite2-City.mmdb') as reader:
            if request.remote_addr == '127.0.0.1':
                user_info.update({
                    'country': 'Local',
                    'city': 'Local',
                    'latitude': 'N/A',
                    'longitude': 'N/A',
                    'timezone': local_timezone,
                    'geoip_error': 'Ejecutando en localhost - solo disponible información local'
                })
            else:
                response = reader.city(request.remote_addr)
                user_info.update({
                    'country': response.country.name,
                    'city': response.city.name,
                    'latitude': response.location.latitude,
                    'longitude': response.location.longitude,
                    'timezone': response.location.time_zone or local_timezone
                })
    except Exception as e:
        user_info.update({
            'country': 'Local',
            'city': 'Local',
            'latitude': 'N/A',
            'longitude': 'N/A',
            'timezone': local_timezone,
            'geoip_error': f'Error de geolocalización: {str(e)}'
        })
    
    return user_info

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/generar_estructura', methods=['POST'])
def generar_estructura():
    try:
        # Crear directorios según el entorno
        user_dir = os.path.join(TEMP_DIR, str(uuid.uuid4()))
        os.makedirs(user_dir, exist_ok=True)
        
        user_static_path = os.path.join(user_dir, 'static')
        os.makedirs(user_static_path, exist_ok=True)
        
        # Configurar logging para este usuario
        logger = setup_logging(user_dir)
        
        # Obtener información del usuario y registrarla
        user_info = get_user_info()
        user_info['action'] = 'generate_structure'
        user_info['canonical_string'] = request.json.get('string_canonico')
        
        # Forzar el flush después de escribir
        logger.info(user_info)
        for handler in logger.handlers:
            handler.flush()
        
        data = request.json
        string_canonico = data.get('string_canonico')
        
        # Interpretar el string canónico
        unidades = interpretar_string_canonico(string_canonico)
        
        print("Iniciando generación de unidades...")
        # Generar cada unidad modificada
        smiles_unidades = []
        for i, unidad in enumerate(unidades):
            print(f"Generando unidad {i+1} de {len(unidades)}...")
            stereo_config = {
                1: 'beta',
                2: 'L',
                3: 'L',
                4: 'D',
                5: 'L'
            }
            stereo_config.update(unidad['stereo_config'])
            
            mol = mol2pdb(
                stereo_config=stereo_config,
                substitution_pattern=unidad['sustituciones'],
                output_file=os.path.join(user_static_path, f"unidad_{i+1}.pdb")
            )
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            smiles_unidades.append(smiles)
        
        print("Iniciando generación de ciclodextrina...", flush=True)
        # Generar la ciclodextrina
        n_units = len(unidades)
        print(f"Número de unidades a generar: {n_units}", flush=True)
        sys.stdout.flush()
        
        paster = GPUPaster(n_units=n_units, output_dir=user_static_path)
        print("GPUPaster inicializado", flush=True)
        sys.stdout.flush()
        
        try:
            print("Iniciando workflow completo de GPUPaster...", flush=True)
            sys.stdout.flush()
            cd_modificada = paster.run()  # El output_dir ya se pasó en el constructor
            print("Workflow de GPUPaster completado", flush=True)
        except Exception as e:
            print(f"Error en GPUPaster: {str(e)}", flush=True)
            raise Exception(f"Error al generar la estructura: {str(e)}")
        
        # Después de generar la estructura
        non_minimized_pdb = os.path.join(user_static_path, 'non_minimized.pdb')
        
        # Esperar a que exista el archivo non_minimized.pdb
        max_wait = 180  # 3 minutos máximo de espera
        start_time = time.time()
        while not os.path.exists(non_minimized_pdb):
            if time.time() - start_time > max_wait:
                raise Exception("Tiempo de espera agotado para la generación de la estructura")
            time.sleep(1)
            print("Esperando generación de archivo PDB...")
        
        print("Archivo PDB generado, procesando estructura...")
        
        # Leer el archivo PDB sin minimizar
        with open(non_minimized_pdb, 'r', encoding='utf-8') as f:
            pdb_data = f.read()
        
        # Verificar si cd_modificada es una molécula válida
        if not isinstance(cd_modificada, Chem.rdchem.Mol):
            cd_modificada = Chem.MolFromPDBFile(non_minimized_pdb)
            if cd_modificada is None:
                raise Exception("No se pudo generar una molécula válida")
        
        # Verificar si la molécula es válida antes de continuar
        if cd_modificada is None:
            raise Exception("No se pudo generar una molécula válida")
            
        # Generar imagen 2D
        try:
            mol_2d = cd_modificada
            AllChem.Compute2DCoords(mol_2d)
        except Exception as e:
            raise Exception(f"Error al generar coordenadas 2D: {str(e)}")
        
        # Mejorar la visualización 2D
        opts = Draw.DrawingOptions()
        opts.bondLineWidth = 3
        opts.atomLabelFontSize = 16
        opts.includeAtomNumbers = False
        opts.additionalAtomLabelPadding = 0.4
        
        # Generar imagen PNG
        img = Draw.MolToImage(mol_2d, size=(1000, 1000), 
            kekulize=True,
            wedgeBonds=True,
            imageType="png",
            fitImage=True,
            options=opts, 
            path=os.path.join(user_static_path, 'estructura.png'))
        
        # Convertir imagen a base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Generar SMILES
        smiles_final = Chem.MolToSmiles(cd_modificada, isomericSmiles=True)
        
        return jsonify({
            'success': True,
            'smiles': smiles_final,
            'imagen_2d': img_str,
            'pdb_noH': pdb_data,
            'pdb_H': pdb_data,
            'mensaje': 'Estructura generada exitosamente',
            'user_dir': user_static_path
        })
        
    except Exception as e:
        # Registrar el error si ocurre
        if 'logger' in locals():
            logger.error(f"Error generating structure: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/minimizar_estructura', methods=['POST'])
def minimizar_estructura():
    try:
        user_dir = request.json.get('user_dir')
        logger = setup_logging(user_dir)
        
        # Obtener información del usuario
        user_info = get_user_info()
        user_info['action'] = 'minimize_structure'
        
        # Registrar la actividad directamente (sin convertir a JSON)
        logger.info(user_info)  # Cambio aquí: ya no usamos json.dumps()
        
        data = request.get_json()
        pdb_data = data.get('pdb_data')
        
        if not pdb_data or not user_dir:
            return jsonify({'success': False, 'error': 'Missing required data'})
        
        if not os.path.exists(user_dir):
            return jsonify({'success': False, 'error': 'Invalid directory'})
        
        # Usar el directorio del usuario para los archivos
        temp_path = os.path.join(user_dir, 'temp_for_min.pdb')
        min_output = os.path.join(user_dir, 'minimized.pdb')
        
        # Crear archivo temporal para la minimización
        with open(temp_path, 'w') as temp_file:
            temp_file.write(pdb_data)
        
        try:
            # Leer la molécula con OpenBabel
            mol = next(pybel.readfile("pdb", temp_path))
            
            # Configurar la minimización
            ff = ob.OBForceField.FindForceField("MMFF94")
            ff.Setup(mol.OBMol)
            
            # Realizar la minimización
            ff.ConjugateGradients(1000)
            ff.GetCoordinates(mol.OBMol)
            
            # Guardar la estructura minimizada
            mol.write("pdb", min_output, overwrite=True)
            
            # Leer el archivo minimizado
            with open(min_output, 'r') as f:
                minimized_pdb = f.read()
            
            # Limpiar archivo temporal
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'minimized_pdb': minimized_pdb
            })
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'success': False,
                'error': f'Minimization error: {str(e)}'
            })
            
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error minimizing structure: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Función para limpiar directorios antiguos periódicamente
def cleanup_old_dirs():
    """Elimina directorios más antiguos que X días"""
    pass  # Implementar según necesidades

if __name__ == '__main__':
    app.run(debug=True, port=5000) 