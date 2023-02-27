import pathlib
import os

root = pathlib.Path(os.path.dirname(__file__)).parent
PROJECT_ROOT_PATH = str(root)

################     CREATE PDFS      #######################################################

playground_folder = 'playground2'
LATEX_PLAYGROUND_PATH = root.joinpath('latex', playground_folder)
LATEX_CONTENT_PATH = root.joinpath('latex', 'content')
LATEX_PDF_RESULTS = root.joinpath('outputs', 'pdfs_to_markup')
LATEX_PDF_RESULTS.mkdir(parents=True, exist_ok=True)
LATEX_MERGED_PDF_RESULTS = root.joinpath('outputs', 'merged_pdfs_to_markup')
LATEX_MERGED_PDF_RESULTS.mkdir(parents=True, exist_ok=True)

################     MARKUP           #######################################################

CLASSES_LIST = '__background__ text section chapter formula img'.split()
CLASSES_NUM = len(CLASSES_LIST)
CLASSNAME_TO_CLASSNUM = dict([(x, i) for i, x in enumerate(CLASSES_LIST)])
CLASSNUM_TO_CLASSNAME = dict([(i, x) for i, x in enumerate(CLASSES_LIST)])
COLORNAME_TO_CLASSNAME = {
    'blue': 'formula',
    'magenta': 'section',
    'yellow': 'section',
    'green': 'text',
    'red': 'text',
    'cyan': 'chapter',
    'black': 'skip',
    'white': '__background__'
}
RGB_TO_COLORNAME = {
    (0, 0, 0): 'black',
    (255, 255, 255): 'white',
    (255, 0, 0): 'red',
    (0, 0, 255): 'blue',
    (0, 255, 0): 'green',
    (0, 255, 255): 'cyan',
    (255, 255, 0): 'yellow',
    (255, 0, 255): 'magenta'
}
CLASSNAME_TO_COLORNAME = {
    'formula': 'blue',
    'section': 'yellow',
    'text': 'red',
    'chapter': 'cyan',
    'img': 'green'
}
CLASSNUM_TO_COLOR = dict([(i, CLASSNAME_TO_COLORNAME[x]) if i > 0 else (None, None) for i, x  in CLASSNUM_TO_CLASSNAME.items()])

################       TRAINING      ###########################################################

DATA_PATH = root.joinpath('outputs', 'data')
TRAINED_MODELS_PATH = root.joinpath('outputs', 'trained_models')
PLOTS_SAVE_PATH = root.joinpath('outputs', 'trained_models')
BATCH_SIZE = 1
NUM_EPOCHS = 20

################        PDF PROCESSING  ###################################################################

PDF_TO_PROCESS = root.joinpath('misc', 'pdf_to_process.pdf')
LATEX_RESULT_PATH = root.joinpath('latex', 'result_here')
IMGS_SAVE_PATH = LATEX_RESULT_PATH.joinpath('files')
IMGS_SAVE_PATH.mkdir(parents=True, exist_ok=True)
#%%
