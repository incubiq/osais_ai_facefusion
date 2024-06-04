
## ------------------------------------------------------------------------
#       Use BASE AI 
## ------------------------------------------------------------------------

import sys
sys.path.insert(0, '../osais_ai_base')

## ensure we have the latest files
from main_init import osais_copyBaseFiles
osais_copyBaseFiles()

## now point to latest source and start APP...
sys.path.remove('../osais_ai_base')
sys.path.insert(0, './_osais')
from main_fastapi import app
