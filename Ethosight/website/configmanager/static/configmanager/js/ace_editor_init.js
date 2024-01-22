window.addEventListener("load", function() {
    var textarea = document.getElementById("id_content");
    var editorDiv = document.createElement("div");
    editorDiv.id = "editor";
    editorDiv.style.height = "400px";
	 editorDiv.style.width = "100%";
    textarea.style.display = "none";
    textarea.parentElement.appendChild(editorDiv);

    var editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");
    editor.session.setMode("ace/mode/yaml");
    editor.setValue(textarea.value);
    editor.setFontSize("14pt");  // Adjust the font size here

    editor.session.on('change', function() {
        textarea.value = editor.getValue();
    });
});

