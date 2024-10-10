$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict2').show();
        $('#result-resnet50').text('');
        $('#result-resnet50').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict2').click(function () {
        var form_data = new FormData($('#upload-file-resnet50')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict_resnet50',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result-resnet50').fadeIn(600);
                $('#result-resnet50').text(' Result:  ' + data.prediction);
                console.log('Success!');
            },
            error: function() {
                $('#result-resnet50').html('Error occurred during prediction.');
            },
        });
    });

});
