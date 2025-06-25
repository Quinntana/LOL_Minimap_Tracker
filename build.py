# build.py
import os
import sys
import PyInstaller.__main__
import shutil

def build_executable():
    # Clean up previous builds
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')

    # PyInstaller configuration with fixes for Scipy
    pyinstaller_args = [
        'main.py',
        '--onefile',
        '--name=LoLChampionTracker',
        '--add-data=overlay.py;.',
        '--add-data=champion_tracker.py;.',
        '--add-data=image_processor.py;.',
        '--add-data=api_client.py;.',
        '--add-data=config.py;.',
        '--add-data=utils.py;.',
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=skimage.metrics',
        '--hidden-import=keyboard',
        '--hidden-import=mss',
        '--hidden-import=cv2',
        '--hidden-import=scipy._cyutility',
        '--hidden-import=scipy.special._cython_special',
        '--hidden-import=scipy.linalg._cythonized_array_utils',
        '--collect-data=skimage',
        '--collect-data=scipy'
    ]

    # Run PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)

if __name__ == '__main__':
    build_executable()


