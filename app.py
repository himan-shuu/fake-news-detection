ValueError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/fake-news-detection/app.py", line 224, in <module>
    main()
    ~~~~^^
File "/mount/src/fake-news-detection/app.py", line 192, in main
    label, confidence = predict_fake_news(article_text, tokenizer, model, input_names)
                        ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/fake-news-detection/app.py", line 102, in predict_fake_news
    prediction = model.predict(inputs_list)
File "/home/adminuser/venv/lib/python3.13/site-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
File "/home/adminuser/venv/lib/python3.13/site-packages/keras/src/layers/input_spec.py", line 186, in assert_input_compatibility
    raise ValueError(
    ...<4 lines>...
    )
