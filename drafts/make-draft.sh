touch $(date +%Y-%m-%d)-draft.md

echo "---" >> $(date +%Y-%m-%d)-draft.md
echo "title:" >> $(date +%Y-%m-%d)-draft.md
echo "excerpt:" >> $(date +%Y-%m-%d)-draft.md
echo "tags:" >> $(date +%Y-%m-%d)-draft.md
echo "  - " >> $(date +%Y-%m-%d)-draft.md
echo "  - " >> $(date +%Y-%m-%d)-draft.md
echo "header:" >> $(date +%Y-%m-%d)-draft.md
echo "  overlay_image: /assets/images/cool-backgrounds/cool-background1.png" >> $(date +%Y-%m-%d)-draft.md
echo "  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'" >> $(date +%Y-%m-%d)-draft.md
echo "last_modified_at:" >> $(date +%Y-%m-%d)-draft.md
echo "---" >> $(date +%Y-%m-%d)-draft.md
