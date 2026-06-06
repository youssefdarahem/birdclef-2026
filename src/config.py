from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent

ALL_CLASSES = [
    '1161364', '116570', '1176823', '1491113', '1595929', '209233', '22930', '22956',
    '22961', '22967', '22973', '22983', '22985', '23150', '23154', '23158',
    '23176', '23724', '24279', '24285', '24287', '24321', '244024', '25073',
    '25092', '25214', '326272', '41970', '43435', '47144', '47158son01', '47158son02',
    '47158son03', '47158son04', '47158son05', '47158son06', '47158son07', '47158son08', '47158son09', '47158son10',
    '47158son11', '47158son12', '47158son13', '47158son14', '47158son15', '47158son16', '47158son17', '47158son18',
    '47158son19', '47158son20', '47158son21', '47158son22', '47158son23', '47158son24', '47158son25', '476521',
    '516975', '517063', '555123', '555145', '555146', '64898', '65377', '65380',
    '66971', '67107', '67252', '70711', '738183', '74113', '74580', '760266',
    'ashgre1', 'astcra1', 'bafcur1', 'baffal1', 'banana', 'barant1', 'batbel1', 'baymac',
    'bbwduc', 'bcwfin2', 'bkcdon', 'bkhpar', 'blchaw1', 'blheag1', 'blttit1', 'bncfly',
    'bobfly1', 'brcmar1', 'brnowl', 'bucmot4', 'bucpar', 'bufpar', 'bunibi1', 'burowl',
    'camfli1', 'chacha1', 'chbmoc1', 'chobla1', 'chvcon1', 'cibspi1', 'coffal1', 'compau',
    'compot1', 'crbthr1', 'crebec1', 'dwatin1', 'epaori4', 'eulfly1', 'fabwre1', 'fepowl',
    'ficman1', 'flawar1', 'fotfly', 'fusfly1', 'gilhum1', 'giwrai1', 'glteme1', 'grasal3',
    'greani1', 'greant1', 'greela', 'grekis', 'grepot1', 'gretho2', 'greyel', 'grfdov1',
    'grhtan1', 'gycwor1', 'horscr1', 'houspa', 'hyamac1', 'larela1', 'lesela1', 'lesgrf1',
    'limpki', 'linwoo1', 'litcuc2', 'litnig1', 'mabpar', 'magant1', 'magtan2', 'masgna1',
    'nacnig1', 'ocecra1', 'oliwoo1', 'orbtro3', 'orwpar', 'osprey', 'pabspi1', 'palhor3',
    'paltan1', 'phecuc1', 'picpig2', 'pirfly1', 'plasla1', 'platyr1', 'plcjay1', 'pluibi1',
    'purjay1', 'pvttyr1', 'ragmac1', 'rebscy1', 'recfin1', 'redjun', 'relser1', 'rinkin1',
    'rivwar1', 'roahaw', 'rubthr1', 'rufcac2', 'rufcas2', 'rufgna3', 'rufhor2', 'rufnig1',
    'ruftho1', 'ruftof1', 'rumfly1', 'ruther1', 'rutjac1', 'sabspa1', 'saffin', 'saytan1',
    'scadov1', 'schpar1', 'scther1', 'shcfly1', 'shshaw', 'shtnig1', 'sibtan2', 'smbani',
    'smbtin1', 'sobcac1', 'sobtyr1', 'socfly1', 'sofspi1', 'souant1', 'soulap1', 'souscr1',
    'spbant3', 'spispi1', 'sptnig1', 'squcuc1', 'stbwoo2', 'strcuc1', 'strher2', 'strowl1',
    'swthum1', 'swtman1', 'tattin1', 'thlwre1', 'toctou1', 'trokin', 'trsowl', 'undtin1',
    'varant1', 'watjac1', 'wesfie1', 'wfwduc1', 'whbant2', 'whbwar2', 'whiwoo1', 'whlspi1',
    'whnjay1', 'whtdov', 'whwpic1', 'y00678', 'yebcar', 'yebela1', 'yecmac', 'yecpar',
    'yehcar1', 'yeofly1',
]

CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
NUM_CLASSES = len(ALL_CLASSES)  # 234


@dataclass
class Config:
    # Paths
    data_dir: Path = ROOT
    train_audio_dir: Path = ROOT / "train_audio"
    train_soundscapes_dir: Path = ROOT / "train_soundscapes"
    test_soundscapes_dir: Path = ROOT / "test_soundscapes"
    train_csv: Path = ROOT / "train.csv"
    taxonomy_csv: Path = ROOT / "taxonomy.csv"
    soundscape_labels_csv: Path = ROOT / "train_soundscapes_labels.csv"
    val_soundscape_labels_csv: Path = ROOT / "train_soundscapes_labels.csv"  # always original, never overridden
    sample_submission_csv: Path = ROOT / "sample_submission.csv"
    checkpoints_dir: Path = ROOT / "checkpoints"
    submissions_dir: Path = ROOT / "submissions"

    # Audio
    sample_rate: int = 32000
    clip_duration: float = 5.0          # seconds per training clip
    clip_samples: int = 160000          # sample_rate * clip_duration

    # Mel spectrogram
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320               # ~10ms at 32kHz → 500 frames per 5s
    f_min: float = 20.0
    f_max: float = 16000.0

    # Dataset mixing
    bird_soundscape_ratio: float = 0.7  # fraction of batches from individual recordings

    # Training
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    warmup_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.01
    val_split: float = 0.2              # fraction of soundscape files held out
    seed: int = 42

    # Augmentation
    mixup_alpha: float = 0.4
    spec_time_mask: int = 40
    spec_freq_mask: int = 20
    noise_snr_db_min: float = 5.0
    noise_snr_db_max: float = 15.0

    # Inference
    inference_overlap: float = 0.0      # 0 = non-overlapping 5s windows (matches submission format)

    # Data quality
    min_rating: float = 0.0              # filter train_audio clips below this rating (0 = keep all)

    # Experiment
    experiment_name: str = "exp001_baseline"
    model_name: str = "efficientnet_b3"
    debug: bool = False
    max_debug_samples: int = 500
