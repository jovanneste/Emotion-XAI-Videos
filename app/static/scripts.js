function showname() {
  var name = document.getElementById('fileInput');
  alert('Selected file: ' + name.files.item(0).name);
  alert('Selected file: ' + name.files.item(0).size);
  alert('Selected file: ' + name.files.item(0).type);
};

function getOption(){
  selectElement = document.querySelector('#select1');
  output = selectElement.value;
  document.querySelector('.output').textContent = output.slice(12);
};
