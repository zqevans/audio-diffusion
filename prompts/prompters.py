import audio_metadata
import random

# A class that generates text prompts from audio metadata
class MetadataPrompter():
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate

    # Creates and returns a text prompt given a metadata object
    def get_track_prompt_from_file_metadata(self, filepath):

        try:
            track_metadata = audio_metadata.load(filepath)
        except Exception as e:
            print(f"Couldn't load metadata for {filepath}: {e}")
            return filepath #If the metadata can't be loaded, use the file path as the prompt

        properties = []

        if "tags" in track_metadata:
            tags = track_metadata["tags"]

            add_property = lambda name, key: random.random() > 0.1 and properties.append(f"{name}: {', '.join(tags[key])}")

            if 'title' in tags:
                add_property("Title", "title")
            if 'artist' in tags:
                add_property("Artist", "artist")
            if 'album' in tags:
                add_property("Album", "album")
            if 'genre' in tags:
                add_property("Genre", "genre")
            if 'label' in tags:
                add_property("Label", "label")
            if 'date' in tags:
                add_property("Date", "date")

        random.shuffle(properties)
        return "|".join(properties)

def get_prompt_from_jmann_metadata(metadata):

    properties = []

    song_attributes = metadata["attributes"]

    attributes = [
        'Date',
        'Location',
        'Topic',
        'Mood',
        'Beard',
        'Genre',
        'Style',
        'Key',
        'Tempo',
        'Year',
        'Instrument'
    ]

    traits = {}

    traits["Name"] = [metadata["name"]]

    traits["Artist"] = ["Jonathan Mann"]

    for attribute in song_attributes:
        trait_type = attribute["trait_type"]
        value = attribute["value"]

        if trait_type not in attributes:
            continue

        if trait_type in traits:
            traits[trait_type].append(value)
        else:
            traits[trait_type] = [value]

    for trait in traits:
        properties.append(f"{trait}: {', '.join(traits[trait])}")

    # Sample a random number of properties
    properties = random.sample(properties, random.randint(1, len(properties)))

    return "|".join(properties)

def get_prompt_from_audio_file_metadata(metadata):

    properties = []

    tags = [
        'title',
        'artist',
        'album',
        'genre',
        'label',
        'date',
        'composer',
        'bpm',
    ]

    for tag in metadata.keys():
        if tag in tags:
            properties.append(f"{tag}: {', '.join(metadata[tag])}")

    if len(properties) == 0:
        if "path" in metadata:
            return metadata["path"]
        elif "text" in metadata:
            return metadata["text"]
        else:
            return ""

    # Sample a random number of properties
    properties = random.sample(properties, random.randint(1, len(properties)))
    #random.shuffle(properties)

    return "|".join(properties)

def get_prompt_from_fma_metadata(metadata):

    properties = []

    keys = ['genre', 'album', 'song_title', 'artist', 'composer']

    if "original_data" in metadata:
        original_data = metadata["original_data"]
        
        for key in keys:
            if key == "song_title":
                prompt_key = "title"
            else:
                prompt_key = key

            if key in original_data and str(original_data[key]) != "nan":
                properties.append(f"{prompt_key}: {original_data[key]}")

    properties = random.sample(properties, random.randint(1, len(properties)))
    return "|".join(properties)