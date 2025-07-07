import random

pokemon_base = [
    ("Bulbasaur", "Bulbasaur es un Pokémon planta/veneno que absorbe energía solar con su bulbo."),
    ("Ivysaur", "Ivysaur es la evolución de Bulbasaur, su flor crece mientras almacena poder."),
    ("Venusaur", "Venusaur es la forma final de Bulbasaur, con una gran flor que emite aromas calmantes."),
    ("Charmander", "Charmander es un lagarto de fuego que usa su cola como fuente de calor."),
    ("Charmeleon", "Charmeleon es más agresivo, con una llama intensa que refleja su temperamento."),
    ("Charizard", "Charizard es un dragón poderoso que puede lanzar llamaradas ardientes desde su boca."),
    ("Squirtle", "Squirtle es una tortuga de agua que dispara potentes chorros para defenderse."),
    ("Wartortle", "Wartortle tiene orejas peludas y una cola esponjosa que simbolizan longevidad."),
    ("Blastoise", "Blastoise lleva cañones de agua en su caparazón para atacar con chorros a presión."),
    ("Caterpie", "Caterpie es una oruga verde que se esconde entre hojas para protegerse."),
    ("Metapod", "Metapod es la forma crisálida de Caterpie, permanece inmóvil para evolucionar."),
    ("Butterfree", "Butterfree es una mariposa colorida que esparce polen paralizante para defenderse."),
    ("Weedle", "Weedle es un gusano con aguijón venenoso que vive entre la hierba."),
    ("Kakuna", "Kakuna es la forma pupa de Weedle, endurece su caparazón para defenderse."),
    ("Beedrill", "Beedrill es una avispa veloz con aguijones dobles para atacar en enjambre."),
    ("Pidgey", "Pidgey es un ave pequeña que vuela bajo y se oculta en arbustos."),
    ("Pidgeotto", "Pidgeotto patrulla su territorio con precisión y gran agudeza visual."),
    ("Pidgeot", "Pidgeot es un ave majestuosa que vuela a gran velocidad para cazar."),
    ("Rattata", "Rattata es un roedor veloz que se adapta a cualquier entorno urbano."),
    ("Raticate", "Raticate tiene colmillos afilados para roer cualquier material duro."),
    ("Spearow", "Spearow es un ave territorial que lanza ataques rápidos para defenderse."),
    ("Fearow", "Fearow sobrevuela vastos territorios y ataca con su pico largo."),
    ("Ekans", "Ekans es una serpiente sigilosa que se enrolla para atacar de sorpresa."),
    ("Arbok", "Arbok intimida enemigos mostrando patrones aterradores en su cuello."),
    ("Pikachu", "Pikachu es un ratón eléctrico que almacena energía en sus mejillas."),
    ("Raichu", "Raichu es la evolución de Pikachu y descarga potentes rayos cuando se enfurece."),
    ("Sandshrew", "Sandshrew se enrolla para protegerse y cava madrigueras subterráneas."),
    ("Sandslash", "Sandslash usa sus púas y garras afiladas para defender su territorio."),
    ("Nidoran♀", "Nidoran♀ es pequeña pero tiene cuernos venenosos para atacar."),
    ("Nidorina", "Nidorina cuida a sus crías y usa su veneno para protegerlas."),
    ("Nidoqueen", "Nidoqueen es fuerte y protectora, defiende a su familia con fiereza."),
    ("Nidoran♂", "Nidoran♂ es ágil y usa sus afilados cuernos venenosos para atacar."),
    ("Nidorino", "Nidorino es territorial y embiste enemigos con su cuerno tóxico."),
    ("Nidoking", "Nidoking es enorme y usa su cola poderosa para aplastar a sus rivales."),
    ("Clefairy", "Clefairy es un Pokémon hada que danza bajo la luz de la luna."),
    ("Clefable", "Clefable es sigilosa y se mueve silenciosa para evitar ser vista."),
    ("Vulpix", "Vulpix es un zorro de fuego con múltiples colas que se agitan al atacar."),
    ("Ninetales", "Ninetales es místico y puede vivir mil años, controla fuego a voluntad."),
    ("Jigglypuff", "Jigglypuff canta melodías que duermen a sus oponentes."),
    ("Wigglytuff", "Wigglytuff tiene un cuerpo suave que se infla para intimidar rivales."),
    ("Zubat", "Zubat es un murciélago que navega en la oscuridad usando ecolocalización."),
    ("Golbat", "Golbat chupa sangre de sus presas para alimentarse durante la noche."),
    ("Oddish", "Oddish absorbe la luz lunar para crecer y esparcir semillas."),
    ("Gloom", "Gloom emite un aroma fétido para mantener alejados a los depredadores."),
    ("Vileplume", "Vileplume tiene un gran pétalo que libera polen tóxico."),
    ("Paras", "Paras cultiva hongos en su espalda que usa para defenderse."),
    ("Parasect", "Parasect está completamente controlado por el hongo que lo habita."),
]

variations = [
    "¿Qué características tiene {name}?",
    "Descríbeme a {name}.",
    "¿Cómo es {name}?",
    "Dime cómo es {name}.",
    "Describe a {name}."
]
output_file = "pokemon_descriptions.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for _ in range(30):  # 30 ciclos para ~1500 ejemplos
        random.shuffle(pokemon_base)  # Mezcla para variar orden
        for name, desc in pokemon_base:
            question = random.choice(variations).format(name=name)
            f.write(f"Pregunta: {question}\nRespuesta: {desc}\n###\n")

print(f"✅ Dataset limpio generado: {output_file}")
