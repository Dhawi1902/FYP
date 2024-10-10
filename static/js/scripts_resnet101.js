$(document).ready(function() {
    // Show preview of image
    $('#imageUpload').change(function() {
        readURL(this);
        $('.image-section').show();
    });

    // Predict
    $('#btn-predict').click(function() {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling the API for ResNet101
        $.ajax({
            type: 'POST',
            url: '/predict_resnet101',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result span').html('Prediction: ' + data.prediction);
                console.log('Success!');
            },
            error: function(data) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result span').html('Error: ' + data.responseJSON.error);
                console.log('Error!');
            }
        });
    });
});

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function(e) {
            $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
            $('#imagePreview').hide();
            $('#imagePreview').fadeIn(650);
        };
        reader.readAsDataURL(input.files[0]);
    }
}
