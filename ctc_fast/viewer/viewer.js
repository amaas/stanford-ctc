$(document).ready(function() {
    table = $('<table>');
    table.attr('id', 'runs');
    table.attr('class', 'tablesorter');

    $.getJSON('data.json', function(data) {
        // Make sure that update time is always there
        $('#last_updated').html(data['time']);
        $('#masthead').html('ctc swbd experiments');
        // First get keys
        var keys = data['keys'];
        // Then populate the table
        thead = $('<thead>');
        tr = $('<tr>');
        $.each(keys, function(ind, val) {
            th = $('<th>');
            th.append(val);
            tr.append(th);
        });
        if (data['figs']) {
            th = $('<th>');
            th.append('fig');
            tr.append(th);
        }
        thead.append(tr);
        tbody = $('<tbody>');
        $.each(data['runs'], function(ind, val) {
            tr = $('<tr>');
            $.each(keys, function(key_ind, key) {
                td = $('<td>');
                if (key in val) {
                    td.append(String(val[key]));
                }
                else {
                    td.append('<span style="color:#aaa">N/A</span>');
                }
                tr.append(td);
            });
            if (data['figs']) {
                td = $('<td>');
                td.html('<a href=\"plots/' + val['run'] + '.png\">plot</a>');
                tr.append(td);
            }
            tbody.append(tr);
        });

        table.append(thead);
        table.append(tbody);

        $('#runs_wrapper').append(table);
        $('#runs').tablesorter({
            widgets: ['zebra'],
            sortList: [[0, 0]]
        });
        $('#runs').trigger('update');
    });

});
