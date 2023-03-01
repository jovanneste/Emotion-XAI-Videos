function showname() {
  var name = document.getElementById('fileInput');
  alert('Selected file: ' + name.files.item(0).name);
  alert('Selected file: ' + name.files.item(0).size);
  alert('Selected file: ' + name.files.item(0).type);
};

function getOption(){
  selectElement = document.querySelector('#select1');
  output = selectElement.value;
  // output var is fake path of video
  document.querySelector('.output').textContent = "0.95";
};

function showFrames(){
  document.getElementById('frame1').style.display = "block";
  document.getElementById('frame2').style.display = "block";
  // document.getElementById('frame3').style.display = "block";
  // document.getElementById('frame4').style.display = "block";
  document.getElementById('frameoutput1').style.display = "block";
  // document.getElementById('frameoutput2').style.display = "block";
  document.getElementById('btnID').style.display = "none";
};
