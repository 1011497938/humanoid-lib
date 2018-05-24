#!/bin/bash
rsync -avz --exclude-from='exclude.txt' ~/humanoid-lib $1:
