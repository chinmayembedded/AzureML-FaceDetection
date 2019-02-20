# Introduction
This project contains the deployment of face recognition module in azure machine learning workspace(preview). `azure_web_service` contains the code files that are required for the deployment of face recognition service.
`face_client` contains the client module which requests the web service to get the recognition results.


# Execution steps


In `azure_web_service`, 
<br />
1. <b>00.configuration.ipynb</b>
<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This file contains the steps for creating the environment along with the resource group and workspace creation. 
<br />
2. <b>face_detection.ipynb</b>
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This file contains the steps for deploying the web service.
<br />
3. After deploying the service we get the <b>scoring_uri</b>. Copy this URI.
<br />

In `face_client`,
<br />
1. <b>config.json</b>
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Paste the scoring URI that is copied from the previous step.
2. <b>request_file.py</b>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This file takes the image url as an input agrument. After executing this file it converts the image provided in the base64 payload and opens a window with image and number of detections.

<br />
### Run command
`python request_file.py --url <image-url>`
<br/>
e.g python request_file.py --url https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/beautycheck/english/durchschnittsgesichter/m(01-32)_gr.jpg

<br/>
Images can be of `.jpg`, `.jpeg`, and `.png` format.