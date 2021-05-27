import re


def isolate_start(rule):
    pattern = re.compile(r"[A-Z]+")
    found = pattern.search(rule)
    return rule[found.start():found.end()]


def main():
    rules_obj = {}

    for rule_block in open("rules.dp").read().split('%'):
        rule = rule_block.strip().split("\n")[1]

        label, tags = rule.split(": ", 1)
        head = isolate_start(tags)
        if head not in rules_obj:
            rules_obj[head] = {}

        rules_obj[head][label] = tags
    return rules_obj


if __name__ == "__main__":
    main()
