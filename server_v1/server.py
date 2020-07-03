from flask import Flask,request
from flask_cors import CORS
from google.cloud import translate
import os
import sys
import json
import base64
import subprocess
sys.path.append('..')
sys.path.append("./i3d-train/experiments/ucf-101")
import test_rgb_flow_result
app = Flask(__name__)
CORS(app)

@app.route("/test", methods=['POST'])
def predict_sign_language():
    
    #save file
    data = json.loads(request.form.get('data'))
    filename = data['filename']
    filedata = data['filedata']    
    filedata=filedata.replace("data:video/webm;base64,", "")
    filedata=base64.b64decode(filedata)
    save_file=open("./orivideo/%s"%(filename),"wb")
    save_file.write(filedata)
    save_file.close()
    
    #ffmpeg
    filename=filename.split('.')[0]
    cmd="sudo ffmpeg -i ./orivideo/%s.webm -r 25 -f mp4 -y ./provideo/%s.mp4"%(filename,filename)
    proc=subprocess.Popen(cmd,shell=True)
    proc.wait()
    
    #denseflow
    dir="./flowvideo/%s"%(filename)
    dir_x="./flowvideo/%s/x"%(filename)
    dir_y="./flowvideo/%s/y"%(filename)
    dir_im="./flowvideo/%s/i"%(filename)
    os.mkdir(dir)
    os.mkdir(dir_x)
    os.mkdir(dir_y)
    os.mkdir(dir_im)
    cmd="sudo ./dense-flow/build/denseFlow_gpu --vidFile='./provideo/%s.mp4' --xFlowFile='%s/flow_x' --yFlowFile='%s/flow_y' --imgFile='%s/im' --bound=16 --type=2 --device_id=0 --step=2"%(filename,dir_x,dir_y,dir_im)
    proc=subprocess.Popen(cmd,shell=True)
    proc.wait()
    print(dir)
    
    #test i3d
    ans=test_rgb_flow_result.run_training(dir)
    
    return ans

@app.route("/test/automl", methods=['POST'])
def callAutoml():
    print "automl"
    print(request.json)
    data=request.json
    print(data["result"])
    translate_client = translate.TranslationServiceClient()
    project_id="805170059749"
    location = 'us-central1'
    model = 'projects/805170059749/locations/us-central1/models/TRL7393067806654201856'
    parent = translate_client.location_path(project_id, location)
    ans = translate_client.translate_text(
        parent=parent,
        contents=[data["result"]],
        model=model,
        mime_type='text/plain',  # mime types: text/plain, text/html
        source_language_code='en',
        target_language_code='zh')
    for translation in ans.translations:
        ans_result = unicode(translation).encode('utf8')
    ans_result = ans_result.split("model",1)
    print(ans_result[0])
    ans_result=ans_result[0].split("\"")
    return json.dumps(ans_result[1])


if __name__ == '__main__':
    cmd="sudo rm -r flowvideo/* provideo/* orivideo/*"
    proc=subprocess.Popen(cmd,shell=True)
    proc.wait()
    test_rgb_flow_result.load_model()
    app.run(host="0.0.0.0",port=1234)
