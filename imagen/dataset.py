import requests
import os
import random
import json
from urllib.parse import urlparse
import time
from pathlib import Path

class PokemonImageDownloader:
    def __init__(self, base_url="https://pokeapi.co/api/v2/pokemon/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PokemonImageDownloader/1.0'
        })
        self.base_folder = "pokemon_images_by_type"
        self.setup_folders()
    
    def setup_folders(self):
        """Crear carpeta base y subcarpetas por tipo"""
        # Crear carpeta principal
        os.makedirs(self.base_folder, exist_ok=True)
        
        # Lista de todos los tipos de Pokémon
        self.pokemon_types = [
            'normal', 'fire', 'water', 'electric', 'grass', 'ice',
            'fighting', 'poison', 'ground', 'flying', 'psychic',
            'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
        ]
        
        # Crear carpetas para cada tipo
        for ptype in self.pokemon_types:
            type_folder = os.path.join(self.base_folder, ptype)
            os.makedirs(type_folder, exist_ok=True)
        
        # Carpeta para tipos duales
        os.makedirs(os.path.join(self.base_folder, 'dual_type'), exist_ok=True)
        
        print(f"✅ Carpetas creadas en: {self.base_folder}")
    
    def get_all_pokemon_ids(self, limit=None):
        """Obtener TODOS los IDs de Pokémon disponibles"""
        try:
            # Obtener total de Pokémon
            response = self.session.get(f"{self.base_url}?limit=1")
            total_pokemon = response.json()['count']
            
            print(f"📊 Total de Pokémon disponibles: {total_pokemon}")
            
            # Si se especifica un límite, usar ese número
            if limit:
                max_pokemon = min(limit, total_pokemon)
                print(f"🎯 Descargando los primeros {max_pokemon} Pokémon")
                return list(range(1, max_pokemon + 1))
            else:
                # Descargar TODOS
                print(f"🌟 Descargando TODOS los {total_pokemon} Pokémon")
                return list(range(1, total_pokemon + 1))
                
        except Exception as e:
            print(f"❌ Error obteniendo lista completa: {e}")
            return []
    
    def get_pokemon_data(self, pokemon_id):
        """Obtener datos completos de un Pokémon"""
        try:
            response = self.session.get(f"{self.base_url}{pokemon_id}")
            if response.status_code == 200:
                data = response.json()
                
                # Extraer información relevante
                pokemon_info = {
                    'id': data['id'],
                    'name': data['name'],
                    'types': [t['type']['name'] for t in data['types']],
                    'sprite_url': data['sprites']['front_default'],
                    'official_artwork_url': None,
                    'height': data['height'],
                    'weight': data['weight']
                }
                
                # Obtener artwork oficial si existe
                if 'other' in data['sprites'] and 'official-artwork' in data['sprites']['other']:
                    artwork = data['sprites']['other']['official-artwork']
                    if artwork and 'front_default' in artwork:
                        pokemon_info['official_artwork_url'] = artwork['front_default']
                
                return pokemon_info
            else:
                print(f"❌ Error HTTP {response.status_code} para Pokémon {pokemon_id}")
                return None
        except Exception as e:
            print(f"❌ Error obteniendo datos del Pokémon {pokemon_id}: {e}")
            return None
    
    def download_image(self, image_url, save_path):
        """Descargar una imagen desde URL"""
        try:
            if not image_url:
                return False
            
            response = self.session.get(image_url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"❌ Error descargando imagen: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error descargando imagen: {e}")
            return False
    
    def save_pokemon_images(self, pokemon_data):
        """Guardar solo el artwork oficial en las carpetas correspondientes"""
        name = pokemon_data['name']
        types = pokemon_data['types']
        artwork_url = pokemon_data['official_artwork_url']

        saved_files = []

        # Determinar carpeta(s) de destino
        if len(types) == 1:
            # Pokémon de un solo tipo
            folders = [os.path.join(self.base_folder, types[0])]
        else:
            # Pokémon de tipo dual - guardar en ambos tipos y en carpeta dual
            folders = [os.path.join(self.base_folder, ptype) for ptype in types]
            folders.append(os.path.join(self.base_folder, 'dual_type'))

        # ✅ SOLO descargar artwork oficial (imagen grande)
        if artwork_url:
            for folder in folders:
                artwork_path = os.path.join(folder, f"{name}_artwork.png")
                if self.download_image(artwork_url, artwork_path):
                    saved_files.append(artwork_path)

        return saved_files
    
    def create_pokemon_info_file(self, pokemon_data, folder_path):
        """Crear archivo JSON con información del Pokémon"""
        info_file = os.path.join(folder_path, f"{pokemon_data['name']}_info.json")
        
        pokemon_info = {
            'id': pokemon_data['id'],
            'name': pokemon_data['name'],
            'types': pokemon_data['types'],
            'height': pokemon_data['height'],
            'weight': pokemon_data['weight'],
            'sprite_url': pokemon_data['sprite_url'],
            'official_artwork_url': pokemon_data['official_artwork_url']
        }
        
        try:
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(pokemon_info, f, indent=2, ensure_ascii=False)
            return info_file
        except Exception as e:
            print(f"❌ Error creando archivo info: {e}")
            return None
    
    def download_all_pokemon(self, limit=None, batch_size=50):
        """Descargar TODOS los Pokémon disponibles"""
        print(f"🚀 Iniciando descarga masiva de Pokémon...")
        
        # Obtener todos los IDs
        pokemon_ids = self.get_all_pokemon_ids(limit)
        if not pokemon_ids:
            print("❌ No se pudieron obtener IDs de Pokémon")
            return [], []
        
        total_pokemon = len(pokemon_ids)
        print(f"📦 Total a descargar: {total_pokemon} Pokémon")
        
        successful_downloads = []
        failed_downloads = []
        
        # Procesar en lotes para mostrar progreso
        for batch_start in range(0, total_pokemon, batch_size):
            batch_end = min(batch_start + batch_size, total_pokemon)
            current_batch = pokemon_ids[batch_start:batch_end]
            
            print(f"\n📦 Procesando lote {batch_start//batch_size + 1} ({batch_start + 1}-{batch_end} de {total_pokemon})")
            
            for i, pokemon_id in enumerate(current_batch):
                current_position = batch_start + i + 1
                
                print(f"📡 [{current_position:4d}/{total_pokemon}] Procesando Pokémon ID: {pokemon_id}")
                
                # Obtener datos del Pokémon
                pokemon_data = self.get_pokemon_data(pokemon_id)
                if not pokemon_data:
                    failed_downloads.append(pokemon_id)
                    print(f"   ❌ Error obteniendo datos")
                    continue
                
                name = pokemon_data['name']
                types = pokemon_data['types']
                
                print(f"   🎯 {name.capitalize()} - Tipos: {', '.join(types)}")
                
                # Descargar imágenes
                saved_files = self.save_pokemon_images(pokemon_data)
                
                # Crear archivos de información
                if len(pokemon_data['types']) == 1:
                    folder = os.path.join(self.base_folder, types[0])
                    self.create_pokemon_info_file(pokemon_data, folder)
                else:
                    for ptype in types:
                        folder = os.path.join(self.base_folder, ptype)
                        self.create_pokemon_info_file(pokemon_data, folder)
                    
                    dual_folder = os.path.join(self.base_folder, 'dual_type')
                    self.create_pokemon_info_file(pokemon_data, dual_folder)
                
                if saved_files:
                    successful_downloads.append({
                        'pokemon': pokemon_data,
                        'files': saved_files
                    })
                    print(f"   ✅ Descargado: {len(saved_files)} archivos")
                else:
                    failed_downloads.append(pokemon_id)
                    print(f"   ❌ Error descargando imágenes")
                
                # Pausa más corta para descarga masiva
                time.sleep(0.2)
            
            # Mostrar progreso del lote
            success_rate = len(successful_downloads) / current_position * 100
            print(f"📊 Progreso del lote: {len(successful_downloads)} exitosos, {len(failed_downloads)} fallidos ({success_rate:.1f}% éxito)")
        
        return successful_downloads, failed_downloads
    
    def generate_summary_report(self, successful_downloads, failed_downloads):
        """Generar reporte resumen"""
        print("\n" + "="*50)
        print("📊 RESUMEN DE DESCARGA")
        print("="*50)
        
        print(f"✅ Descargas exitosas: {len(successful_downloads)}")
        print(f"❌ Descargas fallidas: {len(failed_downloads)}")
        
        if failed_downloads:
            print(f"\n❌ IDs fallidos: {', '.join(map(str, failed_downloads))}")
        
        # Contar por tipos
        type_counts = {}
        for download in successful_downloads:
            for ptype in download['pokemon']['types']:
                type_counts[ptype] = type_counts.get(ptype, 0) + 1
        
        print(f"\n📈 Distribución por tipos:")
        for ptype, count in sorted(type_counts.items()):
            print(f"   {ptype.capitalize()}: {count} Pokémon")
        
        # Mostrar estructura de carpetas
        print(f"\n📁 Estructura de carpetas creada:")
        for root, dirs, files in os.walk(self.base_folder):
            level = root.replace(self.base_folder, '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root)
            if level == 0:
                print(f"{indent}{self.base_folder}/")
            else:
                file_count = len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{indent}{folder_name}/ ({file_count} imágenes)")
    
    def list_downloaded_pokemon(self):
        """Listar todos los Pokémon descargados"""
        print(f"\n🗂️ Pokémon descargados por tipo:")
        
        for ptype in self.pokemon_types:
            type_folder = os.path.join(self.base_folder, ptype)
            if os.path.exists(type_folder):
                images = [f for f in os.listdir(type_folder) if f.endswith('.png')]
                pokemon_names = list(set([f.split('_')[0] for f in images]))
                
                if pokemon_names:
                    print(f"\n🔥 {ptype.upper()} ({len(pokemon_names)} Pokémon):")
                    for name in sorted(pokemon_names):
                        print(f"   - {name.capitalize()}")


def main():
    """Función principal"""
    print("🎮 Descargador COMPLETO de Imágenes de Pokémon")
    print("="*50)
    
    # Crear instancia del descargador
    downloader = PokemonImageDownloader()
    
    try:
        # Preguntar al usuario qué quiere descargar
        print("\n🎯 Opciones de descarga:")
        print("1. Descargar TODOS los Pokémon (puede tomar varias horas)")
        print("2. Descargar los primeros 151 Pokémon (Generación 1)")
        print("3. Descargar los primeros 251 Pokémon (Generaciones 1-2)")
        print("4. Descargar los primeros 386 Pokémon (Generaciones 1-3)")
        print("5. Descargar los primeros 493 Pokémon (Generaciones 1-4)")
        print("6. Descargar los primeros 649 Pokémon (Generaciones 1-5)")
        print("7. Descargar los primeros 721 Pokémon (Generaciones 1-6)")
        print("8. Descargar los primeros 809 Pokémon (Generaciones 1-7)")
        print("9. Descargar los primeros 898 Pokémon (Generaciones 1-8)")
        print("10. Descargar número personalizado")
        
        choice = input("\n🎮 Selecciona una opción (1-10): ").strip()
        
        limit = None
        if choice == "1":
            limit = None  # Todos
            print("🌟 Descargando TODOS los Pokémon...")
        elif choice == "2":
            limit = 151
            print("🔴 Descargando Generación 1 (151 Pokémon)...")
        elif choice == "3":
            limit = 251
            print("🟡 Descargando Generaciones 1-2 (251 Pokémon)...")
        elif choice == "4":
            limit = 386
            print("🔵 Descargando Generaciones 1-3 (386 Pokémon)...")
        elif choice == "5":
            limit = 493
            print("💎 Descargando Generaciones 1-4 (493 Pokémon)...")
        elif choice == "6":
            limit = 649
            print("⚫ Descargando Generaciones 1-5 (649 Pokémon)...")
        elif choice == "7":
            limit = 721
            print("🌈 Descargando Generaciones 1-6 (721 Pokémon)...")
        elif choice == "8":
            limit = 809
            print("☀️ Descargando Generaciones 1-7 (809 Pokémon)...")
        elif choice == "9":
            limit = 898
            print("🗡️ Descargando Generaciones 1-8 (898 Pokémon)...")
        elif choice == "10":
            try:
                limit = int(input("🎯 Ingresa el número de Pokémon a descargar: "))
                print(f"🎲 Descargando {limit} Pokémon...")
            except ValueError:
                print("❌ Número inválido, descargando los primeros 151...")
                limit = 151
        else:
            print("❌ Opción inválida, descargando los primeros 151...")
            limit = 151
        
        # Confirmar descarga
        if limit is None:
            confirm = input("\n⚠️ Esto descargará TODOS los Pokémon (~1300+). ¿Continuar? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Descarga cancelada")
                return
        
        print(f"\n⏰ Iniciando descarga... (Esto puede tomar tiempo)")
        start_time = time.time()
        
        # Descargar Pokémon
        successful, failed = downloader.download_all_pokemon(limit)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Generar reporte
        downloader.generate_summary_report(successful, failed)
        
        # Listar Pokémon descargados
        downloader.list_downloaded_pokemon()
        
        print(f"\n⏰ Tiempo total: {duration/60:.2f} minutos")
        print(f"🎉 ¡Descarga completada!")
        print(f"📁 Revisa la carpeta: {downloader.base_folder}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Descarga interrumpida por el usuario")
        print("📊 Generando reporte de lo descargado hasta ahora...")
        # Aquí podrías agregar código para generar reporte parcial
    except Exception as e:
        print(f"\n❌ Error general: {e}")


if __name__ == "__main__":
    main()