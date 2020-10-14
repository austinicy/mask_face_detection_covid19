(function () {
    "use strict";

    var page = {

        ready: function () {

            var formdata = {};

            $('#uploadFile').fileinput({
                uploadUrl: '/upload',
                theme : 'explorer-fas',
                uploadAsync: false,
                showUpload: false,
                showRemove :true,
                showPreview: true,
                showCancel:true,
                showCaption: true,
                maxFileCount: 1,
                uploadUrl: '/upload',
                allowedFileExtensions: ['jpg', 'png'],
                uploadExtraData: function(previewId, index) {
                    return formdata
                },
                browseClass: "btn btn-primary ",
                dropZoneEnabled: true,
                dropZoneTitle: 'Drag file here！',
            });

            $(".btn-upload-3").on("click", function() {
                var username = $("#username").val();
                if(!username) {
                    alert('Please Enter Name.');
                    return;
                }

                formdata = {
                  "username": $("#username").val()
                }

                $("#uploadFile").fileinput('upload');
            });

            $(".btn-reset-3").on("click", function() {
                $("#username").val('');
                $("#uploadFile").fileinput('clear');
            });

            $('#uploadFile').on('fileuploaded', function(event, data, previewId, index) {
                var fname = data.files[index].name;
                out = '<li>' + 'Uploading File # ' + (index + 1) + ' - ' +
                fname + ' Success' + '</li>';
                $('#kv-success-1 ul').append(out);
                $('#kv-success-1').fadeIn('slow');
            });

            //pending

        }
    }

    $(document).ready(page.ready);

})();


function getContent(type) {
    $.ajax({
        type: 'GET',
        url: '/content_dash',
        data: { 'type': type },
        dataType: 'html',
        success: function (data) {
            $("#main").html(data);
        }
    });
}