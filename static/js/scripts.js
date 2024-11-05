$(document).ready(function() {
    // ResNet50
    $('#btn-predict-resnet50').click(function() {
        var form_data = new FormData($('#upload-file-resnet50')[0]);

        // Make prediction by calling the API
        $.ajax({
            type: 'POST',
            url: '/predict_resnet50',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                $('#result-resnet50').html('Prediction: ' + data.prediction);
            },
            error: function() {
                $('#result-resnet50').html('Error occurred during prediction.');
            }
        });
    });

    // ResNet101
    $('#btn-predict-resnet101').click(function() {
        var form_data = new FormData($('#upload-file-resnet101')[0]);

        // Make prediction by calling the API
        $.ajax({
            type: 'POST',
            url: '/predict_resnet101',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                $('#result-resnet101').html('Prediction: ' + data.prediction);
            },
            error: function() {
                $('#result-resnet101').html('Error occurred during prediction.');
            }
        });
    });

    // JavaScript for changing navbar background on scroll
    window.addEventListener('scroll', function () {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.classList.add('navbar-scrolled');
        } else {
            navbar.classList.remove('navbar-scrolled');
        }
    });

});
