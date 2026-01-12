-- Custom SQL migration file, put your code below! --
UPDATE users SET display_name = username WHERE display_name IS NULL;