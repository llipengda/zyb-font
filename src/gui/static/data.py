data = {
    "common": {
        "style": "padding:5px;color:#ffffff;font-size: 18px;border-width: 1px;border-style: solid;"
                 "border-color: #ffffff;"
    },
    "mainwindow": {
        "TITLE": "Font Creator",
        "bg": "rgb(31,39,72)",
        "ICON": "static/pic/logom.png"
    },
    "menus": {
        "bg": "rgb(54,64,95)",
        "width": 200,
        "radius": "4px"
    },
    "main_logo": {
        "width": 160,
        "height": 80,
        "picpath": "static/pic/logo.png",
        "picpathm": "static/pic/logom.png"
    },
    "folded_logo": {
        "width": 56,
        "height": 80,
    },
    "menu_bg": {
        "width": 170,
        "height": 40,
        "enter": "rgb(81,93,128)",
        "leave": "rgba(81,93,128,0)",
        "press": "rgba(101,113,148,0)"
    },
    "folded_menu_icon": {
        "width": 50,
        "height": 40,
    },
    "menu_icon": {
        "painting": "static/pic/pencil.svg",
        "file": "static/pic/file.svg",
        "setting": "static/pic/setting.svg",
        "about": "static/pic/about.svg",
    },
    "menu_btn": {
        "color": "#ffffff",
        "font": "20px",
        "open": "static/pic/open.svg",
        "close": "static/pic/close.svg"
    },
    "button": {
        "border-width": "2px",
        "bg": "rgb(31,39,72)",
    },
    "topnav": {
        "bg": "rgb(54,64,95)",
        "radius": "4px",
        "width": 200,
        "mini": "static/pic/minus-circle.svg",
        "full": "static/pic/square.svg",
        "close": "static/pic/x-circle.svg",
    },
    "current": {
        "label": "画板",
        "color": "#ffffff"
    },
    "to_btn": {
        "width": 32,
        "height": 32,
        "focusbackground": "rgba(81,93,128,1)",
        "nobackground": "rgba(81,93,128,0)"
    },
    "description": {
        "text": """Version: 1.0.0
Author: 李鹏达 吴泽霖 张耘彪 武泽恺
鸣谢：陈良育老师
        """
    },
    "license": {
        "file": "static/license.txt",
        "style": "QTextEdit {"
                 "   padding: 5px;"
                 "   color: #ffffff;"
                 "   font-size: 18px;"
                 "   border-width: 1px;"
                 "   border-style: solid;"
                 "   border-color: #ffffff;"
                 " }"
                 " QScrollBar:vertical {"
                 "	border: none;"
                 "    background: rgb(52, 59, 72);"
                 "    width: 15px;"
                 "    margin: 21px 0 21px 0;"
                 "	border-radius: 0px;"
                 " }"
                 " QScrollBar::handle:vertical {	"
                 "	background: rgb(190 ,190 ,190);"
                 "    min-height: 25px;"
                 "	border-radius: 4px"
                 " }"
                 " QScrollBar::add-line:vertical {"
                 "     border: none;"
                 "    background: rgb(55, 63, 77);"
                 "     height: 20px;"
                 "	border-bottom-left-radius: 4px;"
                 "    border-bottom-right-radius: 4px;"
                 "     subcontrol-position: bottom;"
                 "     subcontrol-origin: margin;"
                 " }"
                 " QScrollBar::sub-line:vertical {"
                 "	border: none;"
                 "    background: rgb(55, 63, 77);"
                 "     height: 20px;"
                 "	border-top-left-radius: 4px;"
                 "    border-top-right-radius: 4px;"
                 "     subcontrol-position: top;"
                 "     subcontrol-origin: margin;"
                 " }"
                 " QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {"
                 "     background: none;"
                 " }"
                 " QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
                 "     background: none;"
                 " }"
                 " QScrollBar::handle:vertical {	"
                 "	background: rgb(190 ,190 ,190);"
                 "    min-height: 25px;"
                 "	border-radius: 4px"
                 " }"
                 " QScrollBar::handle:vertical:hover {"
                 "    background-color: rgb(211 ,211 ,211);"
                 " }"
                 " QScrollBar::handle:vertical:pressed {"
                 "    background-color: rgb(255 ,250 ,250);"
                 " }"
    },
    "right": [{"title": "画板"},
              {"title": "图库"},
              {"title": "设置"},
              {"title": "关于"}],
    "menu": [{
        "name": "画板",
        "type": "painting"
    }, {
        "name": "图库",
        "type": "file"
    }, {
        "name": "设置",
        "type": "setting"
    }, {
        "name": "关于",
        "type": "about"
    }],
    "show": {
        "style": "color:#ffffff;font-size: 18px;border-width: 1px;border-style: solid;border-color: #ffffff;",
    },
    "draw": {
        "path": "draw",
    },
    "slider": "QSlider::groove:horizontal {"
              "    border-radius: 5px;"
              "    height: 10px;"
              "	margin: 0px;"
              "	background-color: rgb(52, 59, 72);"
              "}"
              "QSlider::groove:horizontal:hover {"
              "	background-color: rgb(55, 62, 76);"
              "}"
              "QSlider::handle:horizontal {"
              "    background-color: rgb(190 ,190 ,190);"
              "    border: none;"
              "    height: 20px;"
              "    width: 20px;"
              "    margin: 0px;"
              "	border-radius: 5px;"
              "}"
              "QSlider::handle:horizontal:hover {"
              "    background-color: rgb(211 ,211 ,211);"
              "}"
              "QSlider::handle:horizontal:pressed {"
              "    background-color: rgb(255 ,250 ,250);"
              "}"
}
