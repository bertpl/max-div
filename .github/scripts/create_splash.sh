#! /bin/sh
# ---------------------------------------------
# Input:  ./images/splash/*
# Output: ./images/splash_with_version.webp
# ---------------------------------------------

# --- check imagemagick version ---
echo "------ ImageMagick version info --------------------------------------------"
magick identify -version
echo "----------------------------------------------------------------------------"

# --- prepare background & blend with logo ---
if [ ! -f ./images/splash/_splash_no_text.png ]; then
    # NOTE: we want to avoid doing these steps in Github Actions, as these take ages to complete...

    # --- prepare background ---
    if [ ! -f ./images/splash/_bg_blended.png ]; then
        echo "Preparing background..."
        magick -size 4864x2304 xc:none \
               -fill black -draw "ellipse 2432,1152 1300,500 0,360" \
               -channel RGBA -blur 0x150 \
               -fill black -draw "rectangle 0,2150 4864,2304" \
               -channel RGBA -blur 0x50 \
               ./images/bg_mask.mpc
        magick ./images/splash/splash_dots.png ./images/bg_mask.mpc -compose over -composite ./images/splash/_bg_blended.png
    fi

    # --- blend background with logo ---
    echo "Blending background with logo..."
    # 1) we blend bg_blended.mpc with splash_text.png using lighten  (-clamp to avoid overflow artifacts)
    # 2) we then blend the result with the original with a certain strength, so we can control the strength of the overall operation
    magick ./images/splash/_bg_blended.png ./images/splash/splash_text.png -compose plus -composite -clamp \
           ./images/splash/splash_text.png -compose blend -define compose:args=50,50 -composite -clamp \
           ./images/splash/_splash_no_text.png

fi

# --- create splash ---
echo "Adding text & version info..."
magick -pointsize 36 -font "./images/splash/google_fonts_montserrat_italic.ttf" "./images/splash/_splash_no_text.png" -gravity SouthWest -fill "#bbbbdd" -annotate +10+5 "DiffusionBee 2.5.3 (FLUX.1-dev + Real-ESRGAN)" "./images/temp.mpc"
magick -pointsize 64 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/temp.mpc" -gravity South -fill "#000000" -annotate +3+22 "Configurable Solver for Maximum Diversity problems with Fairness Constraints." "./images/temp.mpc"
magick -pointsize 64 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/temp.mpc" -gravity South -fill "#eeeeee" -annotate +0+25 "Configurable Solver for Maximum Diversity problems with Fairness Constraints." "./images/temp.mpc"
magick -pointsize 128 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/temp.mpc" -gravity West -fill "black" -annotate +1353+283 "v$(uv version --short)" "./images/temp.mpc"
magick -pointsize 128 -font "./images/splash/google_fonts_montserrat_bold.ttf" "./images/temp.mpc" -gravity West -fill "white" -annotate +1350+280 "v$(uv version --short)" -quality 95 -define webp:lossless=false "./images/splash_with_version.webp"

# --- clean up ---
echo "Cleaning up..."
rm ./images/*.mpc
rm ./images/*.cache
