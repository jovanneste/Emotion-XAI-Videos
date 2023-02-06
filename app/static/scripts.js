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
  document.querySelector('.output').textContent = "Exciting, not funny";
};

function showFrames(){
  document.getElementById('frame').style.display = "block";
  document.getElementById('btnID').style.display = "none";
};
