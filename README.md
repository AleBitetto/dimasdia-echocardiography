# Dimasdia Echocardiography

- inspected all DICOM files to understand color settings, formats and possible color transformation

- 55 DICOM files have problem with RLE encoding
    - ~~best solution is to save DICOM to local PC, extract frames to JPG format and upload again~~
    - JPG format may loose image quality
    - custom function edited from pydicom library seems to fix the problem. No manual conversion needed anymore

- 2 DICOM files have same filename but in different folders. It caused overriding when extracting PNG
    - added an exception to add "_BIS" on filename

- for remaining DICOM files, convesion from YBR to RGB format is needed
    - exported custom function from pydicom library

- inspection of converted frames by visualizing first, middle and last frame
    - so far, 3 DICOM is originally saved in RGB. ~~**Frames need to be converted to grayscale**.~~
    - removed because colordoppler traces blood as well
    
- inspection of differences of frames, saved as GIF
    - possibile solution to isolate area of interest ("triangle")
    
- new data contains a 'PALETTE COLOR' with a single channel image
    - added custom function to convert to RGB

- saved images in numpy array as as pickle

- created custom mask to isolate "triangle"
    - visual check for correct output is perfomed
    
- some files (~100) have identical consecutive frames
    - duplicated frame is removed if this happen on the first frame (~90 files)
    
- created function to make final input for model. Allows to:
    - filter frame size
    - select sampling (first available or equally spaced)
    - select frames or frames difference
    - apply mask



- **issue with images**:
    - ~~annotation around the triangle~~
    - heartbeat signal often overlapping the triangle
    - ~~different width of triangle~~
    - there is a dotted line that occasionally crosses the triangle
    - pay attention on frame rate and check overall time duration. Normalization required?
    - should we isolate one (or more) full cardiac cycle(s)? ECG is not available on all files
