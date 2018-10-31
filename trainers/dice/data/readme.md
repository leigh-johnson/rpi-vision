# License: CC0 (Public Domain)

# Creator: [Mario Lurig](https://mariolurig.com/) 
`mariolurig@gmail.com`

## Image Organization
### ~ 85% / 15% (train / valid)
- all training images are 480x480
- all d4, d8, d10, and d12 validation images are 480x480
- most d6 and d20 validation images are 480x480
- a small percentage of additional d6 and d20 validation images are larger (1024px long side) and completely unlike the training set

## Methodology
_All images were created, edited, and sorted by Mario Lurig._

- Fixed camera positions (minimum 2 angles) used to capture video on a rotating platform with two white lights.
	-  Minimum 5 different dice used on 6 different backgrounds (white and various colors)
	-  Video was then exported as images and then batch cropped to 480x480
- Handheld camera moved over 5+ dice on various wood surfaces (minimum 2) using natural lighting
	- Video edited and exported to images then batch cropped to 480x480
	- Images that were partially out of crop were manually removed

The additional d6 and d20 validation images were from my personal image collection or taken additionally on a variety of surfaces with no care for lighting conditions to work as a more robust test.

The validation images were pulled from the full image set (480x480 images) as a 1/7th slice rather than randomly. If preferred, you could combine train/valid together and randomly assign them via your code; this data organization method was chosen to help beginners.

Finally, images taken in like groups are named in like ways. For instance, d4_angleXXX are all d4 dice taken at an angle. d10_top are all d10 dice taken from the top down. Once again done in an effort to make it easy to add/remove data and see how that changes the results.

_**Note:** There are more d6 and d20 images than d4,d8,d10,d12 due to those two dice being my initial test set before building the rest._