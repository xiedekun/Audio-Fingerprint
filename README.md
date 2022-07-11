# Audio Fingerprint Used by Shazam
<font size =5>

**Reference:** 

[1] M?ller, M. (2015). Fundamentals of music processing: Audio, analysis, algorithms, applications (Vol. 5). Cham: 
Springer.

[2] Wang, A. (2003, October). An industrial strength audio search algorithm. In Ismir (Vol. 2003, pp. 7-13).

[3] A. Olteanu. Gtzan dataset - music genre classification. [Online]. Available: 
https://www.kaggle.com/andradaolteanu/gtzan-dataset-musicgenre-classification

[4] Cohen, L., & Lee, C. (1989, November). Local bandwidth and optimal windows for the short time Fourier 
transform. In Advanced algorithms and architectures for signal processing IV (Vol. 1152, pp. 401-425). 
International Society for Optics and Photonics.

[5] Van der Walt, S., Sch?nberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., ... & Yu, T. 
(2014). scikit-image: image processing in Python. PeerJ, 2, e453.

[6] http://coding-geek.com/how-shazam-works

<font>

## Code

**Jupyter**:
```python
Audio Identification.ipynb 
```

**Python Package**:
```python
audioidentification/audioidentification.py

import audioidentification as aid
targetDir = 'database_recordings'
queryDir = 'query_recordings'
fingerprintDir = 'fingerprint'
output_file = 'output.txt'

aid.fingerprintBuilder(targetDir, fingerprintDir)
aid.audioIdentification(queryDir, fingerprintDir, output_file)
```
**Author**: Dekun Xie

