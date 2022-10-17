# PV-MAT (Panoramic Video - Measurement and Tracking)

The name PV-MAT is an ironic play on The Ideal Gas Law: PV=nRT - There's nothing ideal about the chaotic movement of real-life objects.

PV-MAT creates a panorama out of a video and lets the user make measurements on it and track objects' paths and speeds.

# Demo

![A Demo Video](https://media.giphy.com/media/VxyFqfLxsI4srd5OSV/giphy.gif)

# Installation

To directrly install PV-MAT to your computer choose the appropriate file for your operating system from [this folder](https://mega.nz/folder/ip1HQLLT#3_Qnl3GPnD9Ek6tTnrqAxw).

- **[MacOS:](https://mega.nz/file/e590WI6K#bjzjBUwIyd9jlPEXqiQZQ5MokuQTK9qk7CX9CxIAOMI)** After unzipping the ```PV-MAT for MacOS.zip``` file, you can double click ```PV-MAT.app``` to open. If an *"Unidentified developer"* error message pops up, right click the ```PV-MAT.app``` and choose open. The message should reapear with an ```open``` button this time.

- **[Windows:](https://mega.nz/file/20FhUKpB#6cznQwsbpUj4T1nu9jX3rayu0yAYX4113BUgLxG9L00)** After unzipping the ```PV-MAT for Windows.zip``` file (by right clicking it and selecting *Extract All*), go into the ```PV-MAT``` folder, find and double click the file called ```PV-MAT.exe``` (the ```.exe``` may be missing). If a *"Windows protected your PC"* error message pops up, click *"More info"* and *"Run anyway"*. Sometimes windows justifiably doesn't trust programs downloaded from the internet, and so tries to prevent their opening.

You can [download the demo video here](https://mega.nz/file/f9ERhYaQ#J7wMQrfppweOgWFkCc-vw-aCCHnT5u-d6UhH41NGYnQ), if you would like to tinker with the program. Feel free to test it on any video you would like.

### Build on your machine

If you prefer to clone this project and direclty run it on your machine do the following:

0. If you don't already have Python on your machine, [install it here](https://www.python.org/downloads/).

1. Clone the project or [simply download a zip file](https://github.com/royleibov/PV-MAT/archive/refs/heads/master.zip)

```bash
  git clone https://github.com/royleibov/PV-MAT.git
```

2. Go to the project directory

```bash
  cd PV-MAT
```

3. I would recommend you build a virtual environment in your chosen folder and activate it

- **For windows:**
```cmd
  py -m venv ./
  Scripts\activate.bat
```

- **For MacOS:**
```bash
  python -m venv ./
  source bin/activate
```

4. Install dependencies

```bash
  pip install -r requirements.txt
```

5. Run the app fit for your operating system

```bash
  python PV-MAT\ for\ (Windows/MacOS).py
```

# Usage


# License

[MIT License](LICENSE.txt)

Copyright (c) 2022 Roy Leibovitz
