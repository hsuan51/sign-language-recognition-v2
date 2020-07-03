'use strict'
let inputVideo = document.querySelector('#inputVideo')
let outputVideo = document.querySelector('#outputVideo')

let startBtn = document.querySelector('#startBtn')
let stopBtn = document.querySelector('#stopBtn')
let translateBtn = document.querySelector('#translateBtn')
let clearBtn =document.querySelector('#clearBtn')
let ansBlock =document.getElementById('ans')
let bgColor =document.getElementById('bgColor')

let chunks = []
let constraints = {
  audio: false,
  video: true
}

mediaRecorderSetup()

let mediaRecorder = null
let inputVideoURL = null
let outputVideoURL = null
var count=0
startBtn.addEventListener('click', onStartRecording)
translateBtn.addEventListener('click',translate)
//stopBtn.addEventListener('click', onStopRecording)

function onStartRecording (e) {
  mediaRecorder.start()
  bgColor.classList.remove('bg-info')
  bgColor.classList.add('bg-danger')
  console.log('mediaRecorder.start()')
  setTimeout(function(){
      console.log("record three secends");
      onStopRecording();
  },3000)
}

function onStopRecording (e) {
  mediaRecorder.stop()
  bgColor.classList.remove('bg-danger')
  bgColor.classList.add('bg-info')
  console.log('mediaRecorder.stop()')
}

function mediaRecorderSetup () {
  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      if ('srcObject' in inputVideo) {
        inputVideo.srcObject = stream
      } else {
        inputVideo.src = window.URL.createObjectURL(stream)
      }
      inputVideo.controls = false
      mediaRecorder = new MediaRecorder(stream, {mimeType: 'video/webm;codecs=VP9'})
      mediaRecorder.addEventListener('dataavailable',mediaRecorderOnDataAvailable) 
      mediaRecorder.addEventListener('stop', mediaRecorderOnStop)

      function mediaRecorderOnDataAvailable (e) {
        console.log('mediaRecorder on dataavailable', e.data)
        chunks.push(e.data)
      }

      function mediaRecorderOnStop (e) {
        console.log('mediaRecorder on stop')
        //outputVideo.controls = true
        var blob = new Blob(chunks, { type: 'video/webm' })
        chunks = []
        outputVideoURL = URL.createObjectURL(blob)
        //outputVideo.src = outputVideoURL
        
        var reader = new FileReader();
        reader.readAsDataURL(blob); 
        reader.onloadend = function() {
          var base64data = reader.result;
          var data= {
            data: JSON.stringify({
              'filename': 'test'+count+'.webm',
              'filedata': base64data
            }),
          }
          $.ajax({
            type: 'POST',
            url: '/test',
            data: data
          }).done(function(data) {
            console.log(data);
            ansBlock.innerText +=" "+data;
          });
        }  
        count++;
      }
  })
}

function translate(){
    console.log(ansBlock.innerText);
    const result = JSON.stringify({result:ansBlock.innerText});
    console.log(result);
    $.ajax({
            type: 'POST',
            url: '/test/automl',
            contentType: "application/json;charset=utf-8",
            dataType: "json",
            data: result
          }).done(function(data) {
            console.log(data);
            ansBlock.innerText = data;
          });
}

inputVideo.addEventListener('loadedmetadata', function () {
  inputVideo.play()
  console.log('inputVideo on loadedmetadata')
})

clearBtn.addEventListener('click',function () {
  ansBlock.innerText="";
})
