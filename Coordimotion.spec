# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
    ('data/*', 'data'),
    ('home.png', '.'),
    ('isir-trans.png', '.'),
    ('interjointapp.kv', '.'),
    ('/home/dubois/mp_env/lib/python3.12/site-packages/mediapipe', 'mediapipe/')
    ],
    hiddenimports=['kivy.core.window', 'kivy.core.text', 'kivy.core.image'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Coordimotion',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
