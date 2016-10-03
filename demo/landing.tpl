<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
    <title></title>
    <meta name="author" content="Enrique Manjavacas" />
  </head>

  <body>
    <h1>CharFiller!</h1>
    <p>Add a text to be filled (mark characters to be filled with underscores)</p>
    <form action="/fill" method="POST">
      <textarea type="text" name="text" cols="100" rows="20"
		placeholder="This is a sample _ext to be filled"></textarea>
      <input type="submit" name="save" value="save"></input>
    </form>
    <hr>
      <div>{{text if text else "No input"}}</div>
      <br>
      <div>{{output if output else "No input"}}</div>
  </body>
</html>
