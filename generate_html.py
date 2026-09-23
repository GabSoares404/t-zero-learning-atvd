import base64
import markdown
import re

with open('relatorio_dqn.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

html_content = markdown.markdown(md_content)

def replace_img_with_base64(match):
    img_path = match.group(1)
    try:
        with open(img_path, 'rb') as img_f:
            encoded = base64.b64encode(img_f.read()).decode('utf-8')
        return f'src="data:image/png;base64,{encoded}"'
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return match.group(0)

html_content = re.sub(r'src="(plots/[^"]+\.png)"', replace_img_with_base64, html_content)

html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Relatório DQN</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        img {{ max-width: 100%; height: auto; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

with open('relatorio_dqn_completo.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Relatório HTML com imagens embutidas gerado: relatorio_dqn_completo.html")
