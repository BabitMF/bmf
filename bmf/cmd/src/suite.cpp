#include "../include/menu.h"
#include "../include/trace_config_menu.h"

int main() {
    setlocale(LC_ALL, "");
    initscr();
    curs_set(0);
    keypad(stdscr, TRUE);

    int ch;
    bool terminated = false;
    MenuNavMode nav_mode = NAV_NORMAL;

    Menu root = Menu("Main Menu")
        .Text("BMF 工具集")
        .Text("请选择任务：")
        .SubMenu(Menu("配置Trace工具")
            .Text("配置Trace工具")
            .Text("请选择：")
            .SubMenu(TraceMenu("启动Trace")
                .SetTraceSelection()
                .Text("启动Trace")
                .Text("请选择（按SPACE键选择一个或多个Trace类型，按ENTER确认）：")
                .Option("所有Trace类型", TAG_SELECT_ALL)
                .Option("PROCESSING")
                .Option("SCHEDULE")
                .Option("QUEUE_INFO")
                .Option("INTERLATENCY")
                .Option("THROUGHPUT")
                .Option("CUSTOM")
                .Back("放弃更改，返回上一页")
                .Quit()
            )
            .SubMenu(TraceMenu("禁用Printing")
                .SetPrintDisable()
                .Text("确认禁用Printing，是否要继续？")
                .Text("[YES] 如果想禁用Printing，请按 SPACE，ENTER 或 y。")
                .Text("[NO] 想继续菜单，请按任何其他键。")
                .Confirmation()
            )
            .SubMenu(TraceMenu("禁用Tracelog生成")
                .SetTracelogDisable()
                .Text("确认禁用Tracelog生成，是否要继续？")
                .Text("[YES] 如果想禁用Tracelog生成，请按 SPACE，ENTER 或 y。")
                .Text("[NO] 想继续菜单，请按任何其他键。")
                .Confirmation()
            )
            .SubMenu(TraceMenu("完成配置")
                .SetTraceConfigSave()
                .Text("退出后，在terminal上执行：")
                .Text("$ source env.sh")
                .Text("以上命令行会部署有关Trace的环境变量。\n")
                .Text("按任意键继续")
                .Confirmation()
            )
            .Back()
            .Quit()
        )
        .Quit();

    Menu quit = root.CreateQuitMenu();

    std::stack<Menu*> menu_stack;
    menu_stack.push(&root);
    menu_stack.top()->display();

    while (!terminated) {
        // Await input
        ch=getch();

        if (nav_mode == NAV_NORMAL) {
            switch (ch) {
            // Escape
            case 27:
                menu_stack.push(&quit);
                menu_stack.top()->display();
                nav_mode = NAV_PEND_TERMINATE;
                break;

            case ' ':
                if (menu_stack.top()->menu_type == MENU_MULTI && menu_stack.top()->select_option())
                    break;
            case 10:
            case KEY_ENTER:
                if (menu_stack.top()->get_selected_menu().menu_type == MENU_OPTION) {
                    selected_options = menu_stack.top()->selections;
                    menu_stack.top()->callback();
                    menu_stack.pop();
                } else if (menu_stack.top()->get_selected_menu().menu_type == MENU_BACK) {
                    menu_stack.pop();
                } else {
                    menu_stack.push(&menu_stack.top()->get_selected_menu());
                }
                menu_stack.top()->display();
                if (menu_stack.top()->menu_type == MENU_QUIT)
                    nav_mode = NAV_PEND_TERMINATE;
                else if (menu_stack.top()->menu_type == MENU_CONFIRM)
                    nav_mode = NAV_PEND_CONFIRM;
                selected_options.clear();
                break;
            
            // Up Button
            case KEY_UP:
                menu_stack.top()->prev();
                break;

            // Down Button
            case KEY_DOWN:
                menu_stack.top()->next();
                break;
            
            default:
                break;
            }
        } else {
            if (ch == 'y' || ch == ' ' || ch == 10 || ch == KEY_ENTER) {
                if (nav_mode == NAV_PEND_TERMINATE) {
                    // Terminate program
                    terminated = true;
                } else if (nav_mode == NAV_PEND_CONFIRM) {
                    menu_stack.top()->callback();
                    menu_stack.pop();
                    menu_stack.top()->display();
                }
            } else {
                // Back to program
                menu_stack.pop();
                menu_stack.top()->display();
            }
            nav_mode = NAV_NORMAL;
        }
    }
    
    clear();
    endwin();
    return 0;
}