---
layout: archive
permalink: /files
title: "Files"
author_profile: false
---

<?php
echo "Here are my files";
$path = "./assets/";
$dh = opendir($path);
$i=1;
while (($file = readdir($dh)) !== false) {
    if($file != "." && $file != ".." && $file != "index.php" && $file != ".htaccess" && $file != "error_log" && $file != "cgi-bin") {
        echo "<a href='$path/$file'>$file</a><br /><br />";
        $i++;
    }
}
closedir($dh);
?> 
