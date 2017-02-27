<?php
   $name = array_key_exists("language", $_POST) ? $_POST["language"] : null;  
   if( $name ) {
	echo $name;
        $name = html_entity_decode($name);
	$myfile = fopen("/var/www/html/testfile.txt", "a");
	#or die("Unable to open file!");
	$txt = $name."\n";
	fwrite($myfile, $txt);

	fclose($myfile);
	header("Location: inputtools.html");
      	exit();
   }



?>
<html>
   <body>
   
      <form action = "<?php $_PHP_SELF ?>" method = "POST">
	<textarea name="language">
	</textarea>
         Name: <input type = "text" name = "name1" />
         Age: <input type = "text" name = "age1" />
         <input type = "submit" />
      </form>
   
   </body>
</html>
