''' 
Constants for the project 
'''
from PIL import Image

# Constants saving information
FILES = {}
FRAMES = {}

# Fixed parameters
SAVE_FIG = True
Image.MAX_IMAGE_PIXELS = 1000000000
DPI = 1000
COL_NAMES = ["class_id", "class_name", "x", "y", "width", "height"]
GROUPS = {"BRPE": ["BRPEA", "BRPEC", "BRPEJ", "BRPEW", "BRPEF"],
          "LAGU": ["LAGUA", "LAGUF", "LAGUJ"],
          "BLSK": ["BLSKA", "BLSKF"],
          "MTRN": ["MTRNA", "MTRNF", "AMAVA", "AMOYA"],
          "LGHT": ["GREGA", "GREGC", "GREGF", "SNEGA", "REEGWMA", 
              "WHIBA", "WHIBC", "WHIBN", "WHIBJ", "ROSPA", "TCHEA", "MEGRT",
              "CAEGA", "CAEGF"],
          "DARK": ["GBHEA", "GBHEC", "GBHEJ", "GBHEN", "GBHEE", "GBHEF",
              "REEGA", "REEGF", "REEGC", "TRHEA", "BCNHA", "DCCOA"], 
          "OTHR": ["OTHRA"],
          "TRSH": ["TRASH"],
        }
# I put "REEGA", "REEGF", "REEGC" into "DARK" but I am not sure
# Not sure about "CAEGA", "CAEGF", "MEGRT", "TCHEA", "AMAVA", "DCCOA", "AMOYA"
